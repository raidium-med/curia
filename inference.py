from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoImageProcessor, AutoModel

from dinov2.data.transforms import DinoTransforms
from dinov2.eval import setup
from dinov2.eval.utils import ModelWithIntermediateLayers, check_and_download_model

processor = AutoImageProcessor.from_pretrained(
    "./curia-processor-test", trust_remote_code=True, use_fast=False
)

model = AutoModel.from_pretrained(
    "./curia-model-test",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)
model.to("cuda:0")

# img = np.random.randn(512, 512).astype("float32")
# inputs = processor(images=img, return_tensors="pt")
# inputs = {k: v.to("cuda:0", dtype=model.dtype) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#
# with torch.no_grad():
#     out = model(**inputs)
# emb = out.last_hidden_state[:, 0]

img = np.load("data_downstream/luna16/train/images/03862902a0159_0.npy")
inputs = processor(images=img, return_tensors="pt")
inputs = {
    k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
}

with torch.no_grad():
    out = model(**inputs)
emb = out.last_hidden_state[:, 0]


def get_autocast_dtype(config):
    teacher_dtype_str = (
        config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    )
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


pretrained_weights = check_and_download_model(
    "a79bb490fb5f498bb3a48b63a3a591c2", "474999"
)
config = OmegaConf.load("experiments/a79bb490fb5f498bb3a48b63a3a591c2/config.yaml")
model2 = setup.build_dino_model_for_eval(config, pretrained_weights)
autocast_dtype = get_autocast_dtype(config)
autocast_ctx = partial(
    torch.amp.autocast_mode.autocast,
    enabled=True,
    device_type="cuda",
    dtype=autocast_dtype,
)
config["prediction_size"] = config["crops"]["global_crops_size"]
transforms = DinoTransforms(config, False, False)
transformed = transforms(img)

output = model2(transformed.unsqueeze(0).to("cuda:0"), is_training=True)

model3 = ModelWithIntermediateLayers(model2, 1, autocast_ctx)
model3.to("cuda:0")
output_model3 = model3(transformed.unsqueeze(0).to("cuda:0"))
