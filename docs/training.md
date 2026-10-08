# Training

## Available configurations

Training configurations are stored in `configs/train/`:

| Configuration | Backbone |
| --- | --- |
| `tinyclip.json` | TinyCLIP-auto-ViT-22M-32-Text-10M |
| `openclip.json` | OpenCLIP ViT-B-32 ViT-B-16|
| `mobileclip.json` | MobileCLIP S0 S1|

## Sample dataset

The sample dataset is split into training and validation subsets. Teacher embeddings are provided only for training.

```text
resources/datasets/training/sample/
├── train/
│   ├── images/
│   ├── annotations.txt
│   ├── annotations_text_emb.npy
│   └── annotations_image_emb.npy
└── val/
    ├── images/
    └── annotations.txt
```




## Annotation Format

Multiple captions (num=5) should be provided for the same image by assigning each caption a unique index:

```text
image_001.jpg#0	A red vehicle.
image_001.jpg#1	A parked car.
image_001.jpg#2	A vehicle on the road.
image_001.jpg#3	A red car in traffic.
image_001.jpg#4	A compact automobile.
```

For each training annotation file, the teacher embedding files should share the same base name as the annotation file. For N images and teacher dimension D:

| File                  | Shape     | Row ordering                                                                  |
| --------------------- | --------- | ----------------------------------------------------------------------------- |
| `annotations_text_emb.npy`  | `[5N, D]` | One embedding per caption|
| `annotations_image_emb.npy` | `[N, D]`  | One embedding per unique image|

> **Important:** The annotation entries and teacher embeddings must remain aligned.

## Run training

```bash
# MobileCLIP
python train.py --config configs/train/mobileclip.json --devices 0 --num_workers 4

# TinyCLIP
python train.py --config configs/train/tinyclip.json --devices 0 --num_workers 4

# OpenCLIP
python train.py --config configs/train/openclip.json --devices 0 --num_workers 4
```

## Model Selection

The best checkpoint can be selected using either of the following criteria:

- **Validation loss:** Automatically select the best checkpoint during training.
- **Retrieval performance:** Run `select_best_checkpoints.py` to select the checkpoint with the highest mAP@K.

  ```bash
  python tools/evaluation/select_best_checkpoints.py \
    --config configs/evaluation/mobileclip.json \
    --checkpoint-dir saved/mobileclip_s0/SAFT/version_0/checkpoints \
    --output saved/mobileclip_s0/SAFT/version_0/checkpoint_selection_map_results.json \
    --export-dir saved/mobileclip_s0/SAFT/version_0/best_models \
    --device cuda:0 \
    --batch-size 128 \
    --num-workers 8 \
    --top-k 10
  ```
