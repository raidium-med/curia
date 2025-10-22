<p align="center">
    <img width="654" height="120" alt="Logo horizontal medium_NOIR" src="https://github.com/user-attachments/assets/74c009ba-cb16-4d78-8e04-70efc2960ee5" />
</p>

# Curia - Open Source


This repo is the open source version of Curia by Raidium.

You can access on huggingface:
- The Curia model https://huggingface.co/raidium/curia
- The CuriaBench datasets: https://huggingface.co/datasets/raidium/CuriaBench

## Training

To train a head you need to use the following command:

```bash
uv run dinov2/open_source/trainer.py --config dinov2/open_source/configs/luna16-3D.yaml
```

Using which ever of the configs you want.
To train the trainer uses the API of HuggingFace and HuggingFace datasets

## Inference

Curia's pretrained heads are available on huggingface. 
You can load them and run inference on the benchmark, or on your own datasets.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
model = AutoModelForImageClassification.from_pretrained(
    "raidium/curia, subfolder="luna16-3D", trust_remote_code=True, token=os.environ.get("HF_TOKEN")
)
dataset: DatasetDict = load_dataset("raidium/CuriaBench", "luna16-3D")

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

print(f"Accuracy: {acc / len_dataset:.4f}")
```

The important part is the subfolder which lets you choose which head to use.
