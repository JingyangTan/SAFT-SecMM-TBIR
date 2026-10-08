# Evaluation

Evaluation entry point: `tools/evaluation/evaluate.py`.

## Tasks

| Task | Datasets | Metric |
| --- | --- | --- |
| `map` | SecMM-TBIR  | mAP@K |
| `recall` | MSCOCO Flickr30K | Recall@K |




## Evaluation configurations

| Configuration | Model family |
| --- | --- |
| `configs/evaluation/mobileclip.json` | MobileCLIP |
| `configs/evaluation/tinyclip.json` | TinyCLIP |
| `configs/evaluation/openclip.json` | OpenCLIP |




## Evaluate the released MobileCLIP-S1 sft model

SecMM-TBIR mAP@10:

```bash
python tools/evaluation/evaluate.py \
  --config configs/evaluation/mobileclip.json \
  --task map \
  --finetuned-weight-path resources/models/finetuned/mobileclip_s1/best_overall.pt \
  --top-k 10 \
  --output tmp/evaluation/mobileclip_s1_map.json
```

MSCOCO and Flickr30K Recall@10:

```bash
python tools/evaluation/evaluate.py \
  --config configs/evaluation/mobileclip.json \
  --task recall \
  --finetuned-weight-path resources/models/finetuned/mobileclip_s1/best_overall.pt \
  --top-k 10 \
  --output tmp/evaluation/mobileclip_s1_recall.json
```



## Command-line arguments

| Argument | Meaning |
| --- | --- |
| `--config` | Evaluation configuration file. |
| `--task` | Uses `map` for SecMM-TBIR or `recall` for MSCOCO and Flickr30K. |
| `--finetuned-weight-path` | Loads an exported finetuned `.pt` file. Omit it to evaluate the <strong>foundation model</strong>. |
| `--top-k` | Sets the rank for mAP@K or Recall@K. |


Finetuned evaluation accepts the `.pt` files exported by
`select_best_checkpoints.py` described in [Training](training.md).
