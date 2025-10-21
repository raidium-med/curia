import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from datasets import DatasetDict, load_dataset
from omegaconf import OmegaConf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPooling

from modeling_dinov2 import (
    AttentionConfig,
    AttentionModule,
    Dinov2ForImageClassification,
    Dinov2ForImageClassificationConfig,
    extract_3D_features,
    extract_mask_features,
    pad_features,
)


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, regression=False, attention_cfg=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.regression = regression

        self.attention_module = None
        if attention_cfg:
            attn_config = AttentionConfig(
                num_heads=attention_cfg.get("num_heads", 1),
                num_queries=attention_cfg.get("num_queries", 1),
                use_norm=True,
                use_skip_connection=True,
                attention_block=list(attention_cfg.get("block", ("self", "cross"))),
            )
            self.attention_module = AttentionModule(attn_config, in_dim)

        self.linear = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        if self.attention_module:
            features, _ = self.attention_module(pixel_values)
        else:
            features = pixel_values

        logits = self.linear(features)
        if self.regression:
            logits = logits.squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}


def NumpyToTensor():
    return transforms.Lambda(
        lambda x: x
        if isinstance(x, (torch.Tensor, Image.Image))
        else torch.tensor(x).unsqueeze(0)
    )


class AdaptativeResizeMask(torch.nn.Module):
    """
    Transform to resize the mask, using bilinear interpolation,
    to the target size, with an adaptative threshold, to avoid empty masks.
    First, it tries with an initial threshold. If the mask is empty,
    it uses the maximum value in the resized mask to generate a new threshold.
    """

    def __init__(self, target_size, initial_threshold=0.5):
        super().__init__()
        self.target_size = target_size
        self.initial_threshold = initial_threshold

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        mask_resized = TF.resize(
            x,
            self.target_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = mask_resized > self.initial_threshold
        if mask.sum() == 0:
            new_threshold = torch.max(mask_resized) * 0.5
            mask = mask_resized > new_threshold
        mask = mask.to(dtype=torch.int16)
        return mask.float()


def make_mask_transform(
    crop_size=512,
):
    """
    This is a bilinear interpolation for the mask.
    It can be used only if the mask is a binary mask.
    """
    return transforms.Compose(
        [
            NumpyToTensor(),
            AdaptativeResizeMask((crop_size, crop_size)),
        ]
    )


def _extract_predictions_and_labels(eval_pred):
    """Helper to normalize the output format produced by HF Trainer."""

    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        # The trainer returns a tuple of (logits, loss)
        predictions = predictions[0]

    return predictions, labels


def compute_classification_metrics(eval_pred):
    logits, labels = _extract_predictions_and_labels(eval_pred)
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)

    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    if proba.shape[1] == 2:
        proba = proba[:, 1]

    try:
        auc = roc_auc_score(labels, proba, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "auc": auc}


def compute_regression_metrics(eval_pred):
    predictions, labels = _extract_predictions_and_labels(eval_pred)
    predictions = np.squeeze(np.asarray(predictions))
    labels = np.squeeze(np.asarray(labels))

    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    try:
        r2 = r2_score(labels, predictions)
    except ValueError:
        r2 = float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def scale_lr(learning_rate: float, batch_size: int) -> float:
    # Assuming single-process training, so global size is 1.
    return learning_rate * batch_size / 256.0


def preprocess_function(examples, processor, is_regression):
    images_as_np = [np.array(img, dtype=np.float32) for img in examples["image"]]
    processed_images = processor(images_as_np, return_tensors="pt")
    batch = {"pixel_values": processed_images["pixel_values"]}

    mask_transform = make_mask_transform()
    masks = [
        np.array(mask, dtype=np.uint16) for mask in examples["mask"] if mask is not None
    ]
    if len(masks) > 0:
        batch["mask"] = [mask_transform(mask) for mask in masks]

    label_dtype = torch.float32 if is_regression else torch.long
    batch["labels"] = torch.tensor(examples["target"], dtype=label_dtype)

    return batch


def instantiate_model_and_dataset(
    config, is_regression, train_dataset, val_dataset, test_dataset
):
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    if isinstance(model_cfg, dict):
        model_cfg.setdefault("regression", is_regression)

    model_config = Dinov2ForImageClassificationConfig(**model_cfg)  # type: ignore
    model = Dinov2ForImageClassification(model_config)
    model.dino.requires_grad_(False)
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.classifier.bias)
    processor = AutoImageProcessor.from_pretrained(
        config.model.model_name, trust_remote_code=True
    )

    train_dataset = train_dataset.map(
        partial(preprocess_function, processor=processor, is_regression=is_regression),
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_workers,
    )
    val_dataset = val_dataset.map(
        partial(preprocess_function, processor=processor, is_regression=is_regression),
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_workers,
    )
    test_dataset = test_dataset.map(
        partial(preprocess_function, processor=processor, is_regression=is_regression),
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_workers,
    )

    cols = ["pixel_values", "labels"] + (
        ["mask"] if "mask" in train_dataset.column_names else []
    )
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    return model, train_dataset, val_dataset, test_dataset


def extract_features(examples, processor, backbone, config, is_regression):
    images_as_np = [np.array(img, dtype=np.float32) for img in examples["image"]]
    processed_images = processor(images_as_np, return_tensors="pt")
    pixel_values = processed_images["pixel_values"].cuda()

    mask_transform = make_mask_transform()
    masks = [
        np.array(mask, dtype=np.uint16) for mask in examples["mask"] if mask is not None
    ]
    if len(masks) > 0:
        if len(masks[0].shape) == 2:
            masks = torch.cat([mask_transform(mask) for mask in masks], dim=0)  # type:ignore
        else:
            masks = torch.cat(
                [mask_transform(mask.transpose(2, 0, 1)) for mask in masks], dim=0
            )  # type:ignore

    label_dtype = torch.float32 if is_regression else torch.long
    batch = {"labels": torch.tensor(examples["target"], dtype=label_dtype)}

    is_3d = pixel_values.ndim == 5
    if not is_3d:
        with torch.no_grad():
            outputs = backbone(pixel_values=pixel_values, output_hidden_states=True)

        cls_tokens = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        if len(masks) > 0:
            masks = masks.unsqueeze(1).float()  # type: ignore
            pooled_features = extract_mask_features(
                patch_tokens, masks, use_avgpool=True
            )
        else:
            pooled_features = cls_tokens
    else:
        with torch.no_grad():
            encoder_features = []
            for i in range(pixel_values.shape[0]):
                if pixel_values[i].shape[0] > config.batch_size:
                    output_list = [
                        backbone(
                            pixel_values=pixel_values[i][j : j + config.batch_size]
                        )
                        for j in range(0, pixel_values[i].shape[0], config.batch_size)
                    ]
                    if (
                        config.model.use_n_blocks is None
                        or config.model.use_n_blocks == 1
                    ):
                        hidden_states = tuple(
                            [
                                torch.cat(
                                    [out.last_hidden_state for out in output_list],
                                    dim=0,
                                )
                            ]
                        )
                    else:
                        hidden_states = tuple(
                            [
                                torch.cat(
                                    [out.hidden_states[i] for out in output_list], dim=0
                                )
                                for i in range(len(output_list))
                            ]
                        )
                    output = BaseModelOutputWithPooling(
                        last_hidden_state=torch.cat(
                            [out.last_hidden_state for out in output_list], dim=0
                        ),  # type: ignore
                        hidden_states=hidden_states,  # type: ignore
                    )
                else:
                    output = backbone(
                        pixel_values=pixel_values[i], output_hidden_states=True
                    )

                if len(masks) > 0:
                    features_extracted = extract_3D_features(
                        output,
                        num_slices=config.model.num_slices,
                        mask=masks[i].unsqueeze(0).float(),  # type: ignore
                        use_avgpool_per_slice=config.model.use_avgpool_per_slice,
                        use_avgpool_on_the_volume=config.model.use_avgpool_on_the_volume,
                        use_class_token=config.model.use_class_token,
                    )
                else:
                    features_extracted = extract_3D_features(
                        output,
                        num_slices=config.model.num_slices,
                        kernel_size=config.model.kernel_size,
                        use_n_blocks=config.model.use_n_blocks,
                        use_class_token=config.model.use_class_token,
                        use_avgpool_per_slice=config.model.use_avgpool_per_slice,
                        use_avgpool_on_the_volume=config.model.use_avgpool_on_the_volume,
                    )
                encoder_features.append(
                    features_extracted,
                )

            pooled_features = pad_features(encoder_features)
    batch["pixel_values"] = pooled_features
    return batch


def instantiate_cache_model_and_dataset(
    config, is_regression, train_dataset, val_dataset, test_dataset
):
    backbone = Dinov2Model.from_pretrained(config.model.model_name)
    backbone.cuda()  # type: ignore
    backbone.eval()
    processor = AutoImageProcessor.from_pretrained(
        config.model.model_name, trust_remote_code=True
    )

    is_3D = len(np.array(train_dataset[0]["image"]).shape) > 2
    train_dataset = train_dataset.map(
        partial(
            extract_features,
            processor=processor,
            backbone=backbone,
            config=config,
            is_regression=is_regression,
        ),
        batched=True,
        batch_size=len(train_dataset) if is_3D else config.batch_size,
        num_proc=0,  # Can't use multiple processes with a model on GPU
    )
    val_dataset = val_dataset.map(
        partial(
            extract_features,
            processor=processor,
            backbone=backbone,
            config=config,
            is_regression=is_regression,
        ),
        batched=True,
        batch_size=len(val_dataset) if is_3D else config.batch_size,
        num_proc=0,
    )
    test_dataset = test_dataset.map(
        partial(
            extract_features,
            processor=processor,
            backbone=backbone,
            config=config,
            is_regression=is_regression,
        ),
        batched=True,
        batch_size=len(test_dataset) if is_3D else config.batch_size,
        num_proc=0,
    )

    attention_cfg = OmegaConf.select(config, "model.attention_cfg")
    model = Classifier(
        backbone.config.hidden_size,
        config.model.num_classes,
        regression=is_regression,
        attention_cfg=attention_cfg,
    )
    nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.linear.bias)

    cols = ["pixel_values", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    return model, train_dataset, val_dataset, test_dataset


def main(config):
    dataset: DatasetDict = load_dataset(config.dataset_name, config.dataset_config)  # type: ignore
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    regression_flag = OmegaConf.select(config, "regression")
    is_regression = bool(regression_flag) if regression_flag is not None else False

    use_feature_caching = OmegaConf.select(config, "use_feature_caching")
    if use_feature_caching:
        model, train_dataset, val_dataset, test_dataset = (
            instantiate_cache_model_and_dataset(
                config, is_regression, train_dataset, val_dataset, test_dataset
            )
        )
    else:
        model, train_dataset, val_dataset, test_dataset = instantiate_model_and_dataset(
            config, is_regression, train_dataset, val_dataset, test_dataset
        )

    steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
    max_steps = steps_per_epoch * config.epochs

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_strategy="steps",
        logging_steps=max(10, steps_per_epoch // 10),
        save_strategy="no",
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        report_to="none",
    )

    scaled_lr = scale_lr(config.learning_rate, config.batch_size)
    optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    metrics_fn = (
        compute_regression_metrics if is_regression else compute_classification_metrics
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_fn,
        optimizers=(optimizer, scheduler),  # type: ignore
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    save_head(trainer.model, config.output_dir)  # type: ignore

    print("--- Test Set Evaluation ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset)  # type: ignore
    print(test_results)
    with (Path(config.output_dir) / "test_results.txt").open("w") as f:
        f.write(str(test_results))


def save_head(model, output_dir: str):
    output_path = Path(output_dir) / "head.pt"
    output_path.parent.mkdir(exist_ok=True)

    if isinstance(model, Dinov2ForImageClassification):
        payload = {"classifier": model.classifier.state_dict()}
        if model.attention_module is not None:
            payload["attention"] = model.attention_module.state_dict()
    elif isinstance(model, Classifier):
        payload = {"classifier": model.linear.state_dict()}
        if model.attention_module is not None:
            payload["attention"] = model.attention_module.state_dict()
    else:
        raise TypeError(f"Unknown model type: {type(model)}")

    torch.save(payload, output_path)

    print(f"Saved classifier head to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DINOv2 classifier with OmegaConf config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config.yaml file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    main(config)
