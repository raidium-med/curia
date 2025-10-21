import logging
from pathlib import Path
from typing import Dict, List, Optional

import blosc2
import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_mapping: Dict[str, str] = {
    # "luna16": "luna16",
    # "kits": "kits-mask-predict-anatomy",
    # "emidec-classification-mask": "emidec-classification-mask",
    # "luna16-3D": "luna16-3D",
    # "anatomy-ct": "ts-predict-anatomy-mask-easy",
    # "anatomy-mri": "tsv2-mri-predict-anatomy-mask-pl",
    # "anatomy-ct-hard": "ts-predict-anatomy-mask-hard-pl",
    "abdominal-trauma": "rsna-abdominal-trauma",
    # "atlas-stroke": "ATLASR2-Stroke",
    # "kneeMRI": "kneeMRI-3D",
    # "deep-lesion-site": "deep-lesion-site",
    # "ich": "RSNA-ICH-TRAIN-BALANCED/any",
    # "neural_foraminal_narrowing": "rsna-lumbar-spine/neural_foraminal_narrowing",
    # "spinal_canal_stenosis": "rsna-lumbar-spine/spinal_canal_stenosis",
    # "subarticular_stenosis": "rsna-lumbar-spine/subarticular_stenosis",
    # "ixi": "IXI-AGE-Regression",
    ##"tcia-survival": "tcia-survival-with-time-2.0/radius_1",
    # "covidx-ct": "COVIDxCt_subsample",
    # "oasis": "oasis-1-3D",
}


def create_dataset_dict(dataset_dir: Path) -> DatasetDict:
    """Creates a DatasetDict for a given dataset directory."""
    splits: List[str] = ["train", "val", "test"]
    dataset_dict: Dict[str, Dataset] = {}

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Split '{split}' not found in {dataset_dir}. Skipping.")
            continue

        extra_data_path = split_dir / "extra_data.npy"
        images_dir = split_dir / "images"

        if not extra_data_path.exists() or not images_dir.exists():
            logger.warning(f"Missing extra_data.npy or images directory in {split_dir}. Skipping split.")
            continue

        extra_data: np.ndarray = np.load(extra_data_path, allow_pickle=True)

        data_dict: Dict[str, List] = {field: extra_data[field].tolist() for field in extra_data.dtype.names}

        logger.info(f"Loading {len(extra_data)} images for split {split}...")
        images: List[np.ndarray] = []
        masks: List[Optional[np.ndarray]] = []

        for i, fname in enumerate(extra_data["filename"]):
            if i % 100 == 0:
                logger.info(f"Processing image {i + 1}/{len(extra_data)}")

            img_path = images_dir / fname
            if img_path.exists():
                if str(dataset_dir) == "data_downstream/COVIDxCt_subsample":
                    img = blosc2.load_array(str(img_path))
                else:
                    img = np.load(img_path).astype(np.float32)
                images.append(img)
            else:
                logger.warning(f"Image file {img_path} not found")
                images.append(np.zeros((1, 1), dtype=np.float32))

            fname_path = Path(fname)
            mask_fname = f"{fname_path.stem}.mask{fname_path.suffix}"
            mask_path = images_dir / mask_fname
            if mask_path.exists():
                masks.append(np.load(mask_path))
            else:
                masks.append(None)

        data_dict["image"] = images
        data_dict["mask"] = masks

        logger.info(f"Creating HuggingFace dataset for split {split}...")

        # Create dataset in smaller chunks to avoid memory overflow
        chunk_size = 1000
        datasets = []

        for i in range(0, len(images), chunk_size):
            end_idx = min(i + chunk_size, len(images))
            chunk_data = {field: data_dict[field][i:end_idx] for field in data_dict.keys()}
            chunk_dataset = Dataset.from_dict(chunk_data)
            datasets.append(chunk_dataset)
            logger.info(f"Created chunk {i // chunk_size + 1}/{(len(images) + chunk_size - 1) // chunk_size}")

        # Concatenate all chunks
        if len(datasets) == 1:
            hf_dataset = datasets[0]
        else:
            from datasets import concatenate_datasets

            hf_dataset = concatenate_datasets(datasets)

        dataset_dict[split] = hf_dataset

    return DatasetDict(dataset_dict)


def main() -> None:
    """Main function to upload all datasets."""
    import os

    hf_token: Optional[str] = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

    api: HfApi = HfApi()
    repo_id: str = "raidium/CuriaBench"

    for name, data_dir_name in dataset_mapping.items():
        dataset_dir = Path("data_downstream") / data_dir_name
        logger.info(f"Processing dataset: {name} from {dataset_dir}")

        if not dataset_dir.exists():
            logger.warning(f"Dataset directory {dataset_dir} not found. Skipping.")
            continue

        try:
            dataset_dict = create_dataset_dict(dataset_dir)
            if not dataset_dict:
                logger.warning(f"No data found for dataset {name}. Skipping.")
                continue

            config_name = name.replace("/", "-")

            logger.info(f"Uploading {name} to {repo_id} with config '{config_name}'...")
            dataset_dict.push_to_hub(repo_id, config_name=config_name, token=hf_token, max_shard_size="1GB")
            logger.info(f"Successfully uploaded {name}.")

        except Exception as e:
            logger.error(f"Failed to process and upload {name}: {e}")
            continue


if __name__ == "__main__":
    main()
