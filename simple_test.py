import os

from trainer import make_mask_transform
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

repo_id = "raidium/curia"

processor = AutoImageProcessor.from_pretrained(
    repo_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
)
backbone = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(
    repo_id,
    subfolder="luna16-3D",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
dataset: DatasetDict = load_dataset("raidium/CuriaBench", "luna16-3D")  # type: ignore

acc = 0
len_dataset = len(dataset["test"])
for i in range(len_dataset):
    img = np.array(dataset["test"][i]["image"])
    processed = processor(images=img, return_tensors="pt")
    mask_transform = make_mask_transform()
    mask = dataset["test"][i].get("mask", None)
    if mask is not None:
        mask = np.array(mask)
        if mask.ndim == 3:
            processed["mask"] = (
                mask_transform(mask.transpose(2, 0, 1)).transpose(1, 3).transpose(1, 2)
            )
        else:
            processed["mask"] = mask_transform(
                torch.tensor([dataset["test"][i].get("mask", None)])
            ).unsqueeze(0)

    output = model(**processed)
    target = dataset["test"][i]["target"]
    output_class = output["logits"].argmax(-1).item()
    acc += target == output_class
    print(acc / (i + 1))

print(f"Accuracy: {acc / len_dataset:.4f}")
