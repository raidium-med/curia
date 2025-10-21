import argparse
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from modeling_dinov2 import (
    Dinov2ForImageClassification,
    Dinov2ForImageClassificationConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Upload a classifier head to Hugging Face Hub"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config.yaml file for the head",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory for the head",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name for the Hugging Face repository",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please set the HF_TOKEN environment variable."
        )

    config = OmegaConf.load(args.config)

    temp_dir = Path("./temp_upload")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model_config = Dinov2ForImageClassificationConfig(**model_cfg)
    model = Dinov2ForImageClassification(model_config)

    classifier_path = Path(args.model_path) / "head.pt"
    if classifier_path.exists():
        state_dict = torch.load(classifier_path, map_location="cpu")
        model.classifier.load_state_dict(state_dict["classifier"])
        if model.attention_module:
            model.attention_module.load_state_dict(state_dict["attention"])
        print("Loaded classifier weights")

    model.save_pretrained(temp_dir)
    model_config.save_pretrained(temp_dir)

    api = HfApi()
    api.upload_folder(
        folder_path=str(temp_dir),
        path_in_repo=str(args.config).split("/")[-1].split(".")[0],
        repo_id=args.repo_name,
        token=hf_token,
        commit_message="Upload classifier head",
    )

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
