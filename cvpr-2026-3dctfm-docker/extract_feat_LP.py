#!/usr/bin/env python3
"""Feature extraction using Curia (2D DINOv2-based model) for CT volumes.

Curia processes individual axial slices in PL orientation (512x512, patch_size=16).
- Non-ROI: all slices processed, CLS token pooled across slices.
- ROI: each masked slice is center-cropped around the mask (min 128x128),
  resized to 512x512, and only patch tokens overlapping with the mask
  are averaged. Features are then mean-pooled across slices.
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import math
import os

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def reorient_to_lps(image_sitk):
    """Reorient to LPS so axial slices are in PL orientation."""
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image_sitk)


# ---------------------------------------------------------------------------
# ROI crop helpers
# ---------------------------------------------------------------------------

def crop_around_mask(slice_2d, mask_2d, min_size=128):
    """Square center-crop around mask centroid.

    Crop side = max(min_size, mask_bbox_height, mask_bbox_width).
    Clamped to image boundaries.
    """
    h, w = slice_2d.shape
    ys, xs = np.where(mask_2d > 0)

    if len(ys) == 0:
        # No foreground – return full slice / mask
        return slice_2d.copy(), mask_2d.copy()

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    cy = int(ys.mean())
    cx = int(xs.mean())

    crop_size = max(min_size, bbox_h, bbox_w)

    y_start = max(0, cy - crop_size // 2)
    x_start = max(0, cx - crop_size // 2)
    y_end = y_start + crop_size
    x_end = x_start + crop_size

    # Clamp to image boundaries
    if y_end > h:
        y_end = h
        y_start = max(0, h - crop_size)
    if x_end > w:
        x_end = w
        x_start = max(0, w - crop_size)

    return (
        slice_2d[y_start:y_end, x_start:x_end].copy(),
        mask_2d[y_start:y_end, x_start:x_end].copy(),
    )


def resize_mask_to_input(mask_2d, target_size=512):
    """Resize a 2D binary mask to (target_size, target_size) with nearest interp.

    Returns tensor of shape (1, 1, target_size, target_size).
    """
    t = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(target_size, target_size), mode="nearest")


# ---------------------------------------------------------------------------
# Masked token pooling
# ---------------------------------------------------------------------------

def masked_token_average(patch_tokens, masks_512):
    """Average patch tokens that overlap with the mask.

    Args:
        patch_tokens: (B, num_patches, dim)  – CLS token already removed.
        masks_512:    (B, 1, 512, 512)       – binary mask at input resolution.

    Returns:
        (B, dim) tensor – one feature vector per sample.
    """
    B, num_patches, dim = patch_tokens.shape
    spatial_dim = int(math.sqrt(num_patches))  # 32 for Curia
    patch_tokens = patch_tokens.view(B, spatial_dim, spatial_dim, dim)

    # Down-sample mask to patch grid with max-pool (any overlap → 1)
    kernel_size = masks_512.shape[-1] // spatial_dim  # 512 // 32 = 16
    masks_ds = F.max_pool2d(
        masks_512, kernel_size=kernel_size, stride=kernel_size
    )  # (B, 1, spatial_dim, spatial_dim)
    masks_ds = masks_ds.permute(0, 2, 3, 1)  # (B, spatial_dim, spatial_dim, 1)

    # Weighted average over spatial dims
    denom = masks_ds.sum(dim=(1, 2)).clamp(min=1)  # (B, 1)
    features = (patch_tokens * masks_ds).sum(dim=(1, 2)) / denom  # (B, dim)
    return features


# ---------------------------------------------------------------------------
# Feature extraction – non-ROI path
# ---------------------------------------------------------------------------

def extract_features_no_roi(model, processor, volume, device, batch_size):
    """Process all axial slices; average CLS tokens across volume.

    Returns a (dim,) vector (e.g. 768).
    """
    num_slices = volume.shape[0]
    slices = [volume[z].astype(np.float32) for z in range(num_slices)]

    all_cls = []
    for i in range(0, num_slices, batch_size):
        batch = slices[i : i + batch_size]
        inputs = processor(batch)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)

        cls_tokens = outputs.last_hidden_state[:, 0]  # (B, dim)
        all_cls.append(cls_tokens.cpu())

    return torch.cat(all_cls, dim=0).mean(dim=0)  # (dim,)


# ---------------------------------------------------------------------------
# Feature extraction – ROI path
# ---------------------------------------------------------------------------

def extract_features_roi(
    model, processor, volume, mask_volume, device, batch_size, min_crop=128
):
    """Center-crop each masked slice, forward through Curia, average only
    patch tokens overlapping with the mask, then mean-pool across slices.
    """
    slice_indices = [
        z for z in range(volume.shape[0]) if mask_volume[z].any()
    ]
    if not slice_indices:
        # Fallback: use all slices with full patch-token average (keep 768-dim)
        print("  Warning: mask is empty – averaging all patch tokens over all slices")
        num_slices = volume.shape[0]
        slices = [volume[z].astype(np.float32) for z in range(num_slices)]
        all_patch_avg = []
        for i in range(0, num_slices, batch_size):
            batch = slices[i : i + batch_size]
            inputs = processor(batch)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            all_patch_avg.append(outputs.last_hidden_state[:, 1:].mean(dim=1).cpu())
        return torch.cat(all_patch_avg, dim=0).mean(dim=0)  # (768,)

    all_features = []

    for i in range(0, len(slice_indices), batch_size):
        batch_indices = slice_indices[i : i + batch_size]

        cropped_slices = []
        masks_512 = []

        for z in batch_indices:
            crop_s, crop_m = crop_around_mask(
                volume[z].astype(np.float32), mask_volume[z], min_size=min_crop
            )
            cropped_slices.append(crop_s)
            masks_512.append(resize_mask_to_input(crop_m, target_size=512))

        # Processor resizes crops to 512x512 & normalises
        inputs = processor(cropped_slices)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Stack masks → (B, 1, 512, 512)
        masks_batch = torch.cat(masks_512, dim=0).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Patch tokens (drop CLS at position 0)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # (B, 1024, 768)

        # Average only tokens overlapping with the mask
        feats = masked_token_average(patch_tokens, masks_batch)  # (B, 768)
        all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)  # (num_slices, dim)
    return all_features.mean(dim=0)  # (dim,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curia feature extraction for CT volumes"
    )
    parser.add_argument(
        "-i", "--input", dest="imgs_path", type=str,
        default="/workspace/inputs",
        help="Path to input images directory (.nii.gz files)",
    )
    parser.add_argument(
        "-o", "--output", dest="dest", type=str,
        default="/workspace/outputs",
        help="Destination folder for output .h5 feature files",
    )
    parser.add_argument(
        "--masks_path", type=str, default=None,
        help="Path to foreground masks for ROI diseases (optional)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Number of slices per forward pass",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of classes (kept for interface compatibility)",
    )
    parser.add_argument(
        "--model_path", type=str, default="/opt/app/model",
        help="HuggingFace model ID or local path to Curia model",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    print(f"Loading Curia model from {args.model_path}...")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")

    os.makedirs(args.dest, exist_ok=True)

    # Find input images
    supported_extensions = (".nii.gz", ".nii", ".nrrd", ".mha")
    imgs_files = sorted(
        f for f in os.listdir(args.imgs_path)
        if any(f.endswith(ext) for ext in supported_extensions)
    )
    if args.masks_path:
        imgs_files = [
            f for f in imgs_files
            if os.path.exists(os.path.join(args.masks_path, f))
        ]

    print(f"Found {len(imgs_files)} images to process")

    for img_file in tqdm(imgs_files, desc="Processing volumes"):
        if img_file.endswith(".nii.gz"):
            img_id = img_file[: -len(".nii.gz")]
        else:
            img_id = os.path.splitext(img_file)[0]

        img_path = os.path.join(args.imgs_path, img_file)

        # Load & reorient to LPS (axial slices → PL)
        image_sitk = sitk.ReadImage(img_path)
        image_sitk = reorient_to_lps(image_sitk)
        volume = sitk.GetArrayFromImage(image_sitk)  # (Z, Y, X) = (S, P, L)

        # ROI or non-ROI path
        if args.masks_path:
            mask_path = os.path.join(args.masks_path, img_file)
            mask_sitk = sitk.ReadImage(mask_path)
            mask_sitk = reorient_to_lps(mask_sitk)
            mask_volume = sitk.GetArrayFromImage(mask_sitk)

            embedding = extract_features_roi(
                model, processor, volume, mask_volume, device, args.batch_size
            )
        else:
            embedding = extract_features_no_roi(
                model, processor, volume, device, args.batch_size
            )

        # Save as .h5
        out_path = os.path.join(args.dest, f"{img_id}.h5")
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("y_hat", data=embedding.numpy())

        torch.cuda.empty_cache()

    print(
        f"Done! Processed {len(imgs_files)} volumes. "
        f"Features saved to {args.dest}"
    )


if __name__ == "__main__":
    main()
