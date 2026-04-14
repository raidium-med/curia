# Curia Docker for CVPR 2026 3D CT Foundation Models Competition

Docker container for feature extraction using the [Curia](https://huggingface.co/raidium/curia) foundation model, for the [CVPR 2026 Foundation Models for 3D CT](https://github.com/kmin940/CVPR26-3DCTFMCompetition) competition (Task 1 - Linear Probing).

## Build

1. Download the Curia model weights into a `model/` directory:

```bash
mkdir model
# Copy from a local clone of raidium/curia:
cp /path/to/curia/{config.json,model.safetensors,preprocessor_config.json,curia_image_processor.py,modeling_dinov2.py} model/
```

2. Build the Docker image:

```bash
docker build -t curia_lp:latest .
```

## Usage

### Non-ROI diseases

```bash
docker container run --gpus "device=0" -m 32G --name curia_lp --rm \
  -v $PWD/inputs/:/workspace/inputs/ \
  -v $PWD/outputs/:/workspace/outputs/ \
  curia_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
```

### ROI diseases (e.g., adrenal_hyperplasia)

```bash
docker container run --gpus "device=0" -m 32G --name curia_lp --rm \
  -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia \
  -v $PWD/inputs/:/workspace/inputs/ \
  -v $PWD/outputs/:/workspace/outputs/ \
  curia_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
```

### Save as tar.gz for submission

```bash
docker save curia_lp:latest | gzip > curia_lp.tar.gz
```

## Feature extraction details

- **Non-ROI**: Processes all axial slices of each NIfTI volume, extracts CLS token features, and averages across slices. Output: 768-dim vector per volume.
- **ROI**: Center-crops each masked slice around the mask (min 128x128), resizes to 512x512, and averages only patch tokens overlapping with the mask. Output: 768-dim vector per volume.
- Volumes are reoriented to LPS so axial slices are in PL orientation as required by Curia.
- Output format: `.h5` files with key `y_hat`.
