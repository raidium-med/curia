from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ChannelDimension

# at top of the file, add a compatible import
try:
    from transformers.image_processing_utils import BatchFeature
except Exception:
    from transformers.feature_extraction_utils import BatchFeature


_BICUBIC = Image.BICUBIC


class CuriaImageProcessor(BaseImageProcessor):
    """
    1-channel medical preprocessor replicating:
      NumpyToTensor -> float32 -> Resize(crop_size, BICUBIC, antialias)
      -> optional ClipIntensity(min=-1000) -> NormalizeIntensity(channel_wise=True)
    Outputs: pixel_values as (B, 1, crop_size, crop_size)

    Images needs to be in:

    - PL for axial
    - IL for coronal
    - IP for sagittal

    for CT, no windowing, just hounsfield or normalized image
    for MRI, similar, no windowing, just raw values or normalized image
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        crop_size: int = 512,
        clip_below_air: bool = False,
        eps: float = 1e-6,
        do_resize: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crop_size = int(crop_size)
        self.clip_below_air = bool(clip_below_air)
        self.eps = float(eps)
        self.do_resize = bool(do_resize)
        self.do_normalize = bool(do_normalize)

    @staticmethod
    def _move_slices_first(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return array[None, ...]
        if array.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, received shape {array.shape}")
        slice_axis = int(np.argmin(array.shape))
        return np.moveaxis(array, slice_axis, 0)

    def _to_volume_tensor(
        self, image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> torch.Tensor:
        """Returns a tensor shaped (num_slices, H, W)."""
        if isinstance(image, Image.Image):
            if image.mode not in ("L", "F"):
                image = image.convert("L")
            array = np.array(image)
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            array = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        volume = self._move_slices_first(array)
        return torch.from_numpy(volume).float()

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize a tensor shaped (S, H, W) to (S, crop_size, crop_size) using bicubic interpolation.
        If do_resize is False, returns the input tensor unchanged.
        """
        if not self.do_resize:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected tensor shaped (slices, H, W); got {tensor.shape}"
            )
        tensor = tensor.unsqueeze(1)  # (S,1,H,W)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.crop_size, self.crop_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return tensor[:, 0]

    def _clip_min(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.clip_below_air:
            torch.clamp_min(tensor, -1000.0, out=tensor)
        return tensor

    def _zscore_per_image(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            std = tensor.std(dim=(1, 2), keepdim=True)
            std = torch.where(std < self.eps, torch.ones_like(std), std)
            return (tensor - mean) / std
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < self.eps:
            return tensor - mean
        return (tensor - mean) / std

    @staticmethod
    def _compute_slice_indices(
        total_slices: int, target_slices: Optional[int]
    ) -> np.ndarray:
        if target_slices is None or target_slices <= 0:
            return np.arange(total_slices, dtype=int)
        if total_slices >= target_slices:
            return np.linspace(0, total_slices - 1, target_slices, dtype=int)
        pad = np.full(target_slices - total_slices, total_slices - 1, dtype=int)
        return np.concatenate([np.arange(total_slices, dtype=int), pad])

    def __call__(
        self, images, return_tensors="pt", data_format=ChannelDimension.FIRST, **kwargs
    ):
        if not isinstance(images, (list, tuple)):
            images = [images]

        target_slices: Optional[int] = kwargs.pop("num_slices", None)
        slice_indices_override = kwargs.pop("slice_indices", None)
        slice_valid_override = kwargs.pop("slice_valid_mask", None)

        batch = []
        slice_masks: list[np.ndarray] = []
        collected_indices: list[np.ndarray] = []
        all_single_slice = True

        for idx, img in enumerate(images):
            volume = self._to_volume_tensor(img)  # (S, H, W)
            original_len = volume.shape[0]

            override_indices = None
            if slice_indices_override is not None:
                override_indices = slice_indices_override[idx]
                if override_indices is not None:
                    override_indices = np.asarray(override_indices, dtype=np.int64)

            if override_indices is not None:
                indices = np.clip(override_indices, 0, max(original_len - 1, 0))
            else:
                indices = self._compute_slice_indices(original_len, target_slices)

            if indices.shape[0] != 1:
                all_single_slice = False

            volume = volume[indices]

            override_mask = None
            if slice_valid_override is not None:
                override_mask = slice_valid_override[idx]
                if override_mask is not None:
                    override_mask = np.asarray(override_mask, dtype=np.bool_)

            if override_mask is not None and override_mask.shape[0] == indices.shape[0]:
                slice_mask = override_mask
            else:
                valid_len = min(original_len, indices.shape[0])
                slice_mask = np.zeros(indices.shape[0], dtype=np.bool_)
                slice_mask[:valid_len] = True

            volume = volume.to(torch.float32)
            volume = self._resize(volume)
            volume = self._clip_min(volume)
            if self.do_normalize:
                volume = self._zscore_per_image(volume)
            volume = volume[:, None, ...]  # (S,1,H,W)

            batch.append(volume.numpy())
            slice_masks.append(slice_mask)
            collected_indices.append(indices.astype(np.int32))

        pixel_values = np.stack(batch, axis=0)

        if all_single_slice:
            pixel_values = pixel_values[:, 0]
            data: Dict[str, Any] = {"pixel_values": pixel_values}
        else:
            data = {"pixel_values": pixel_values}
            if slice_masks:
                lengths = {mask.shape[0] for mask in slice_masks}
                if len(lengths) == 1:
                    data["slice_mask"] = np.stack(slice_masks, axis=0)
            if collected_indices:
                lengths = {indices.shape[0] for indices in collected_indices}
                if len(lengths) == 1:
                    data["slice_indices"] = np.stack(collected_indices, axis=0)

        return BatchFeature(data=data, tensor_type=return_tensors)

    # saved as preprocessor_config.json
    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update(
            dict(
                crop_size=self.crop_size,
                clip_below_air=self.clip_below_air,
                eps=self.eps,
                do_resize=self.do_resize,
                do_normalize=self.do_normalize,
            )
        )
        # Make AutoImageProcessor discoverable
        out["auto_map"] = {
            "AutoImageProcessor": "curia_image_processor.CuriaImageProcessor"
        }
        return out
