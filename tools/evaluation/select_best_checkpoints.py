"""Select the best checkpoint on SecMM-TBIR and export inference weights.

For detailed usage, see ``docs/training.md``.
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import torch

if __package__:
    from .model_loading import (
        build_mobileclip_model,
        initialize_clip_foundation_model,
    )
    from .retrieval_utils import (
        evaluate_map,
        save_evaluation_results,
        validate_map_evaluation_config,
    )
else:
    from model_loading import (
        build_mobileclip_model,
        initialize_clip_foundation_model,
    )
    from retrieval_utils import (
        evaluate_map,
        save_evaluation_results,
        validate_map_evaluation_config,
    )


CHECKPOINT_NAME_PATTERN = re.compile(
    r"^epoch=(?P<epoch>\d+)-step=(?P<step>\d+)-"
    r"val_loss=(?P<val_loss>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\.ckpt$"
)
DEFAULT_CHECKPOINT_DIR = (
    "saved/tinyclip_10M/SAFT/version_24/checkpoints"
)
DEFAULT_SELECTION_OUTPUT_PATH = (
    "saved/tinyclip_10M/SAFT/version_24/"
    "checkpoint_selection_map_results.json"
)
DEFAULT_EXPORT_DIR = "saved/tinyclip_10M/SAFT/version_24/best_models"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_TOP_K = 10

def parse_evaluation_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/evaluation/tinyclip.json",
        help="Path to the model evaluation config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the Lightning .ckpt files to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_SELECTION_OUTPUT_PATH,
        help="Path to the checkpoint selection result JSON file.",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Directory for the selected finetuned .pt weights.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of evaluation data-loading workers.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Rank cutoff used for checkpoint selection with mAP@K.",
    )
    return parser.parse_args(args)


def _parse_checkpoint_path(checkpoint_path):
    match = CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename does not match the expected training format: "
            f"{checkpoint_path.name}"
        )
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "epoch": int(match.group("epoch")),
        "global_step": int(match.group("step")),
        "val_loss": float(match.group("val_loss")),
    }


def discover_checkpoints(checkpoint_dir):
    checkpoint_paths = list(Path(checkpoint_dir).glob("epoch=*.ckpt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No epoch checkpoints found in checkpoint directory: {checkpoint_dir}"
        )
    checkpoints = [_parse_checkpoint_path(path) for path in checkpoint_paths]
    checkpoints.sort(key=lambda item: (item["epoch"], item["global_step"]))
    return checkpoints


def _load_lightning_state_dict(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")
    return checkpoint["state_dict"]


def _load_clip_lightning_checkpoint(model, checkpoint_path):
    checkpoint_state_dict = _load_lightning_state_dict(checkpoint_path)
    state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("model.model."):
            fixed_key = key.removeprefix("model.model.")
        elif key.startswith("model."):
            fixed_key = key.removeprefix("model.")
        else:
            raise RuntimeError(f"Unexpected checkpoint parameter prefix: {key}")
        state_dict[fixed_key] = value

    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    print(f"missing keys: {sorted(missing_keys)}")
    print(f"unexpected keys: {sorted(unexpected_keys)}")
    if missing_keys or unexpected_keys - {"image_proj.weight"}:
        raise RuntimeError("Finetuned checkpoint does not match the CLIP encoders.")
    model.load_state_dict(state_dict, strict=False)


def _load_mobileclip_lightning_checkpoint(model, checkpoint_path):
    checkpoint_state_dict = _load_lightning_state_dict(checkpoint_path)
    state_dict = {
        key.replace("model.", "", 1): value
        for key, value in checkpoint_state_dict.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )
    print(f"missing keys: {missing_keys}")
    print(f"unexpected keys: {unexpected_keys}")
    return missing_keys, unexpected_keys


def _initialize_lightning_checkpoint_model(model_config, checkpoint_path):
    if model_config["type"] == "mobileclip":
        from mobileclip.modules.common.mobileone import reparameterize_model

        model, preprocess, tokenizer = build_mobileclip_model(
            model_config["architecture"],
            model_config["model_config_path"],
        )
        _load_mobileclip_lightning_checkpoint(model, checkpoint_path)
        model = reparameterize_model(model)
        return model, preprocess, tokenizer

    model, preprocess, tokenizer = initialize_clip_foundation_model(
        model_config["type"],
        model_config["architecture"],
        model_config["pretrained_path"],
    )
    _load_clip_lightning_checkpoint(model, checkpoint_path)
    return model, preprocess, tokenizer


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _export_inference_weights(model_config, checkpoint_path, output_path):
    """Export an unwrapped, CPU state dict from a Lightning checkpoint."""
    if model_config["type"] == "mobileclip":
        model, _preprocess, _tokenizer = build_mobileclip_model(
            model_config["architecture"],
            model_config["model_config_path"],
        )
        missing_keys, unexpected_keys = _load_mobileclip_lightning_checkpoint(
            model,
            checkpoint_path,
        )
        if missing_keys or set(unexpected_keys) - {"image_proj.weight"}:
            raise RuntimeError(
                "Finetuned checkpoint does not match the MobileCLIP encoders."
            )
    else:
        model, _preprocess, _tokenizer = _initialize_lightning_checkpoint_model(
            model_config,
            checkpoint_path,
        )

    state_dict = _cpu_state_dict(model)
    invalid_keys = [
        key
        for key in state_dict
        if key.startswith("model.") or key == "image_proj.weight"
    ]
    if invalid_keys:
        raise RuntimeError(
            f"Inference state dict contains wrapper parameters: {invalid_keys[:5]}"
        )
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError("Inference state dict must contain tensors only.")

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, output_path)
    return output_path


def load_evaluation_config(config_path, checkpoint_dir, evaluation_config):
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)

    for field in ("model", "benchmarks"):
        if field not in config:
            raise ValueError(f"Missing evaluation config field: {field}")
    model = config["model"]
    model_type = model.get("type")
    if model_type not in {"tinyclip", "openclip", "mobileclip"}:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")

    required_fields = ["architecture", "pretrained_path"]
    if model_type == "mobileclip":
        required_fields.append("model_config_path")
    for field in required_fields:
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"Invalid evaluation config field: model.{field}")

    if not os.path.isfile(model["pretrained_path"]):
        raise FileNotFoundError(
            f"Configured model file does not exist: {model['pretrained_path']}"
        )
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("Checkpoint directory must be a non-empty string.")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )
    if model_type == "mobileclip" and not os.path.isfile(
        model["model_config_path"]
    ):
        raise FileNotFoundError(
            f"Configured model file does not exist: {model['model_config_path']}"
        )

    validate_map_evaluation_config(
        {
            "evaluation": evaluation_config,
            "benchmarks": config["benchmarks"].get("map"),
        }
    )
    return config


def _is_better_checkpoint(candidate, current_best):
    if current_best is None:
        return True
    if candidate["mean_map_at_k"] != current_best["mean_map_at_k"]:
        return candidate["mean_map_at_k"] > current_best["mean_map_at_k"]
    if candidate["val_loss"] != current_best["val_loss"]:
        return candidate["val_loss"] < current_best["val_loss"]
    return (candidate["epoch"], candidate["global_step"]) < (
        current_best["epoch"],
        current_best["global_step"],
    )


def _update_best_checkpoint(output, checkpoint_result):
    candidate = {
        "checkpoint_path": checkpoint_result["checkpoint_path"],
        "epoch": checkpoint_result["epoch"],
        "global_step": checkpoint_result["global_step"],
        "val_loss": checkpoint_result["val_loss"],
        "top_k": checkpoint_result["benchmarks"][0]["top_k"],
        "mean_map_at_k": checkpoint_result["mean_map_at_k"],
        "benchmark_map_at_k": {
            result["name"]: result["map_at_k"]
            for result in checkpoint_result["benchmarks"]
        },
    }
    if _is_better_checkpoint(candidate, output["best_overall"]):
        output["best_overall"] = candidate


def _export_best_checkpoint(model_config, export_dir, output):
    export_dir = Path(export_dir)
    best_result = output["best_overall"]
    output_path = export_dir / "best_overall.pt"
    _export_inference_weights(
        model_config,
        best_result["checkpoint_path"],
        output_path.as_posix(),
    )
    best_result["inference_weight_path"] = output_path.as_posix()
    output["selected_model"] = dict(best_result)
    print(f"Exported inference weights: {output_path.as_posix()}", flush=True)


def run_checkpoint_selection(
    config,
    checkpoint_dir,
    export_dir,
    evaluation_config,
):
    model_config = config["model"]
    evaluation_config = dict(evaluation_config)
    checkpoints = discover_checkpoints(checkpoint_dir)
    evaluation_device = evaluation_config["device"]

    output = {
        "model": model_config,
        "evaluation": {
            "device": evaluation_device,
            "batch_size": evaluation_config["batch_size"],
            "num_workers": evaluation_config["num_workers"],
            "top_k": evaluation_config["top_k"],
            "selection_metric": "mean_map_at_k",
            "selection_mode": "max",
            "tie_breakers": ["val_loss", "epoch", "global_step"],
            "selection_benchmarks": [
                benchmark["name"] for benchmark in config["benchmarks"]["map"]
            ],
        },
        "best_overall": None,
        "checkpoints": [],
    }

    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        checkpoint_path = checkpoint["checkpoint_path"]
        print(
            f"Evaluating checkpoint {checkpoint_index}/{len(checkpoints)}: "
            f"{checkpoint_path}",
            flush=True,
        )
        model, preprocess, tokenizer = _initialize_lightning_checkpoint_model(
            model_config,
            checkpoint_path,
        )
        model.to(evaluation_device)
        model.eval()

        checkpoint_result = {**checkpoint, "benchmarks": []}
        for benchmark in config["benchmarks"]["map"]:
            result, _top_indices = evaluate_map(
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                query_file=benchmark["query_file"],
                image_root=benchmark["image_root"],
                label_file=benchmark["label_file"],
                evaluation_device=evaluation_device,
                batch_size=evaluation_config["batch_size"],
                num_workers=evaluation_config["num_workers"],
                top_k=evaluation_config["top_k"],
            )
            benchmark_result = {"name": benchmark["name"], **result}
            checkpoint_result["benchmarks"].append(benchmark_result)
            print(
                f"{benchmark['name']}: mAP@{result['top_k']}="
                f"{result['map_at_k']}",
                flush=True,
            )

        checkpoint_result["mean_map_at_k"] = sum(
            result["map_at_k"] for result in checkpoint_result["benchmarks"]
        ) / len(checkpoint_result["benchmarks"])
        print(
            f"mean mAP@{evaluation_config['top_k']}="
            f"{checkpoint_result['mean_map_at_k']}",
            flush=True,
        )
        output["checkpoints"].append(checkpoint_result)
        _update_best_checkpoint(output, checkpoint_result)
        save_evaluation_results(evaluation_config["result_file"], output)

        del model, preprocess, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _export_best_checkpoint(model_config, export_dir, output)
    save_evaluation_results(evaluation_config["result_file"], output)

    best_result = output["selected_model"]
    print("Selected model:", flush=True)
    for benchmark_name, map_at_k in best_result["benchmark_map_at_k"].items():
        print(
            f"{benchmark_name}: mAP@{best_result['top_k']}={map_at_k}",
            flush=True,
        )
    print(
        f"mean mAP@{best_result['top_k']}={best_result['mean_map_at_k']}, "
        f"checkpoint={best_result['checkpoint_path']}, "
        f"inference_weight={best_result['inference_weight_path']}",
        flush=True,
    )
    return output


def main(args=None):
    parsed_args = parse_evaluation_args(args)
    evaluation_config = {
        "device": parsed_args.device,
        "batch_size": parsed_args.batch_size,
        "num_workers": parsed_args.num_workers,
        "top_k": parsed_args.top_k,
        "result_file": parsed_args.output,
    }
    if not parsed_args.export_dir:
        raise ValueError("Export directory must be a non-empty string.")
    config = load_evaluation_config(
        parsed_args.config,
        parsed_args.checkpoint_dir,
        evaluation_config,
    )
    run_checkpoint_selection(
        config,
        parsed_args.checkpoint_dir,
        parsed_args.export_dir,
        evaluation_config,
    )


if __name__ == "__main__":
    main()
