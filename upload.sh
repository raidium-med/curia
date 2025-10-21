#!/bin/bash
run_id=$1
step=$2

uv run dinov2/open_source/save_model.py --run_id "$run_id" --step "$step"
uv run dinov2/open_source/save_processor.py --run_id "$run_id"
cp dinov2/open_source/curia_image_processor.py curia/
uv run upload_hf.py
