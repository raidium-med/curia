from typing import Any, Dict, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import convert_image_dtype
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

    def _to_tensor(
        self, image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> torch.Tensor:
        """Accepts (H,W), (1,H,W) or PIL; returns torch.float32 tensor (H, W) in grayscale."""
        if isinstance(image, Image.Image):
            # force single channel
            if image.mode != "L" and image.mode != "F":
                image = image.convert("L")
            arr = np.array(image)
            tensor = torch.from_numpy(arr)
            return tensor.float()

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected 2D grayscale tensor or (1,H,W); got shape {tensor.shape}"
                )
            return tensor.float()

        if isinstance(image, np.ndarray):
            arr = image
            # squeeze singleton channel dim if present
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D grayscale array or (1,H,W); got shape {arr.shape}"
                )
            tensor = torch.from_numpy(arr)
            return tensor.to(torch.int16)

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize a 2D torch.Tensor (H, W) to (crop_size, crop_size) using bicubic interpolation.
        If do_resize is False, returns the input tensor unchanged.
        """
        if not self.do_resize:
            return tensor
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor (H, W), got shape {tensor.shape}")
        # Add batch and channel dimensions: (1,1,H,W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.crop_size, self.crop_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        # Remove batch and channel dimensions: (crop_size, crop_size)
        return tensor[0, 0]

    def _clip_min(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.clip_below_air:
            torch.clamp_min(tensor, -1000.0, out=tensor)
        return tensor

    def _zscore_per_image(self, tensor: torch.Tensor) -> torch.Tensor:
        # channel-wise=True with 1 channel -> per image z-score
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < self.eps:
            # avoid exploding when image is constant; center only
            return tensor - mean
        return (tensor - mean) / std

    def __call__(
        self, images, return_tensors="pt", data_format=ChannelDimension.FIRST, **kwargs
    ):
        if not isinstance(images, (list, tuple)):
            images = [images]

        batch = []
        for img in images:
            if len(img.shape) == 3:
                full_volume = []
                for i in range(img.shape[-1]):
                    x = self._to_tensor(img[:, :, i])
                    x = convert_image_dtype(x, torch.float32)
                    x = self._resize(x)
                    x = self._clip_min(x)  # optional
                    x = x[None, ...]
                    full_volume.append(x)
                x = torch.stack(full_volume, dim=0)
                x = self._zscore_per_image(x)  # per-image z-score
            else:
                x = self._to_tensor(img)
                x = convert_image_dtype(x, torch.float32)
                x = self._resize(x)
                x = self._clip_min(x)  # optional
                x = self._zscore_per_image(x)  # per-image z-score
                x = x[None, ...]  # -> (1,H,W)
            batch.append(x)

        pixel_values = np.stack(batch, axis=0)  # (B,1,H,W)

        # 🔧 replace the old self.to_tensor(...) with this:
        return BatchFeature(
            data={"pixel_values": pixel_values},
            tensor_type=return_tensors,  # "pt" | "np" | "tf" | "jax" | None
        )

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
