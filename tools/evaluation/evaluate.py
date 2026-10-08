"""Evaluate foundation or fine-tuned models on retrieval benchmarks.

For detailed usage, see ``docs/evaluation.md``.
"""

import argparse
import json
import os
from pathlib import Path

if __package__:
    from .model_loading import (
        initialize_foundation_model,
        initialize_inference_model,
    )
    from .retrieval_utils import (
        run_map_evaluation,
        run_recall_evaluation,
        validate_map_evaluation_config,
        validate_recall_evaluation_config,
    )
else:
    from model_loading import (
        initialize_foundation_model,
        initialize_inference_model,
    )
    from retrieval_utils import (
        run_map_evaluation,
        run_recall_evaluation,
        validate_map_evaluation_config,
        validate_recall_evaluation_config,
    )


TASK_CHOICES = ("map", "recall")
DEFAULT_OUTPUT_PATH = "tmp/evaluation/evaluation_results.json"
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
        default="configs/evaluation/mobileclip.json",
        help="Path to the model evaluation config.",
    )
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="map",
        help=(
            "Evaluation task: map uses SecMM-TBIR pedestrian/vehicle; "
            "recall uses MSCOCO/Flickr30K."
        ),
    )
    parser.add_argument(
        "--finetuned-weight-path",
        default=None,
        help=(
            "The default None evaluates the "
            "pretrained/foundation model."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the evaluation result JSON file.",
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
        help="Rank cutoff used by mAP@K or Recall@K.",
    )
    return parser.parse_args(args)


def _require_non_empty_string(mapping, field, location):
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid evaluation config field: {location}.{field}")
    return value


def _get_benchmarks(config, task):
    try:
        benchmarks = config["benchmarks"][task]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing evaluation config field: benchmarks.{task}"
        ) from error
    return benchmarks


def load_evaluation_config(
    config_path,
    task,
    evaluation_config,
    finetuned_weight_path=None,
):
    if task not in TASK_CHOICES:
        raise ValueError(f"Unsupported evaluation task: {task}")
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)

    for field in ("model", "benchmarks"):
        if field not in config:
            raise ValueError(f"Missing evaluation config field: {field}")

    model = config["model"]
    model_type = model.get("type")
    if model_type not in {"tinyclip", "openclip", "mobileclip"}:
        raise ValueError(f"Unsupported evaluation model type: {model_type}")
    _require_non_empty_string(model, "architecture", "model")
    pretrained_path = _require_non_empty_string(model, "pretrained_path", "model")
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Configured model file does not exist: {pretrained_path}"
        )
    if model_type == "mobileclip":
        model_config_path = _require_non_empty_string(
            model,
            "model_config_path",
            "model",
        )
        if not os.path.isfile(model_config_path):
            raise FileNotFoundError(
                f"Configured model file does not exist: {model_config_path}"
            )

    if finetuned_weight_path is not None:
        if not isinstance(finetuned_weight_path, str) or not finetuned_weight_path:
            raise ValueError(
                "Finetuned weight path must be a non-empty string when provided."
            )
        if not os.path.isfile(finetuned_weight_path):
            raise FileNotFoundError(
                f"Finetuned weight file does not exist: {finetuned_weight_path}"
            )
        if Path(finetuned_weight_path).suffix.lower() != ".pt":
            raise ValueError(
                "Finetuned weight file must be an exported .pt file: "
                f"{finetuned_weight_path}"
            )

    benchmarks = _get_benchmarks(config, task)
    run_config = {
        "evaluation": evaluation_config,
        "benchmarks": benchmarks,
    }
    if task == "map":
        validate_map_evaluation_config(run_config)
    else:
        validate_recall_evaluation_config(run_config)
    return config


def run_evaluation(
    config,
    task,
    evaluation_config,
    finetuned_weight_path=None,
):
    if finetuned_weight_path is None:
        model, preprocess, tokenizer = initialize_foundation_model(config["model"])
        weight_source = "foundation"
    else:
        model, preprocess, tokenizer = initialize_inference_model(
            config["model"],
            finetuned_weight_path,
        )
        weight_source = "finetuned"

    benchmarks = _get_benchmarks(config, task)
    model_metadata = {**config["model"], "weight_source": weight_source}
    if finetuned_weight_path is not None:
        model_metadata["finetuned_weight_path"] = finetuned_weight_path

    runner = run_map_evaluation if task == "map" else run_recall_evaluation
    return runner(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        model_metadata=model_metadata,
        evaluation_config=evaluation_config,
        benchmarks=benchmarks,
    )


def main(args=None):
    parsed_args = parse_evaluation_args(args)
    evaluation_config = {
        "device": parsed_args.device,
        "batch_size": parsed_args.batch_size,
        "num_workers": parsed_args.num_workers,
        "top_k": parsed_args.top_k,
        "result_file": parsed_args.output,
    }
    config = load_evaluation_config(
        parsed_args.config,
        parsed_args.task,
        evaluation_config,
        parsed_args.finetuned_weight_path,
    )
    run_evaluation(
        config,
        parsed_args.task,
        evaluation_config,
        parsed_args.finetuned_weight_path,
    )


if __name__ == "__main__":
    main()
