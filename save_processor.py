import argparse

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoImageProcessor

from dinov2.open_source.curia_image_processor import CuriaImageProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, default=None, help="Run ID to load config from"
    )
    args = parser.parse_args()

    config = OmegaConf.load(f"experiments/{args.run_id}/config.yaml")

    processor = CuriaImageProcessor(
        crop_size=int(config.crops["global_crops_size"]),
        clip_below_air=bool(config["crops"].get("clip_below_air", False)),
    )

    # save (creates preprocessor_config.json with auto_map)
    processor.auto_map = {
        "AutoImageProcessor": "curia_image_processor.CuriaImageProcessor"
    }
    processor.save_pretrained("./curia")

    proc2 = CuriaImageProcessor.from_pretrained("./curia")

    # NOTE: to make it work you need to cp curia_image_processor.py to ./curia
    proc3 = AutoImageProcessor.from_pretrained("./curia", trust_remote_code=True)

    img = np.random.randn(512, 512).astype("float32")  # 1-channel
    batch = processor(images=img, return_tensors="pt")
    assert batch["pixel_values"].shape == (
        1,
        1,
        processor.crop_size,
        processor.crop_size,
    )
    print(processor)
    print(batch)
