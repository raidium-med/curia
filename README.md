# Curia - Open Source

This repo is the open source version of Curia by Raidium

## Training

To train a head you need to use the following command:

```bash
uv run dinov2/open_source/trainer.py --config dinov2/open_source/configs/luna16-3D.yaml
```

Using which ever of the configs you want.
To train the trainer uses the API of HuggingFace and HuggingFace datasets

## Upload to HuggingFace

To upload $datasets$ to HuggingFace simply run `upload_dataset.py` script

For the $heads$ however, once you train you need to specify:

- the config you used
- the path to the model you trained
- the repo name you want to upload to
Like so:

```bash
uv run dinov2/open_source/upload_heads_to_hf.py \
--config dinov2/open_source/configs/luna16-3D.yaml \
--model_path results/luna16-3D --repo_name raidium/test-curia
```

## Inference

Once you have uploaded your datasets and your head you can do inference like so:

```python
processor = AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
backbone = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(
    repo_id, subfolder="luna16-3D", trust_remote_code=True, token=os.environ.get("HF_TOKEN")
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

print(f"Accuracy: {acc / len_dataset:.4f}")
```

The important part is the subfolder which lets you choose which head to use.
