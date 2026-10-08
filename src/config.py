"""Validation for training JSON configuration."""

import math


def validate_config(config):
    """Reject invalid configuration without changing its values."""
    def fail(field, requirement):
        raise ValueError(f"Invalid config: {field} {requirement}.")

    def required(field):
        value = config
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                fail(field, "is required")
            value = value[key]
        return value

    def mapping(field):
        value = required(field)
        if not isinstance(value, dict):
            fail(field, "must be an object")
        return value

    def string(field):
        value = required(field)
        if not isinstance(value, str) or not value.strip():
            fail(field, "must be a non-empty string")
        return value

    def positive_integer(field):
        value = required(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            fail(field, "must be a positive integer")

    def boolean(field):
        value = required(field)
        if not isinstance(value, bool):
            fail(field, "must be a boolean")
        return value

    def number(field, positive=False):
        value = required(field)
        requirement = "must be finite and > 0" if positive else "must be finite and >= 0"
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            fail(field, requirement)
        return value

    def string_list(field):
        value = required(field)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            fail(field, "must be a non-empty list of non-empty strings")
        return value

    def positive_pair(field, maximum=None):
        value = required(field)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item <= 0
                for item in value
            )
            or value[0] > value[1]
        ):
            fail(field, "must be an ordered pair of finite positive numbers")
        if maximum is not None and value[1] > maximum:
            fail(field, f"must not exceed {maximum}")

    if not isinstance(config, dict):
        fail("root", "must be an object")
    for field in (
        "arch",
        "arch.args",
        "dataset",
        "dataset.args",
        "optimizer",
        "training",
        "scheduler",
        "loss",
    ):
        mapping(field)
    optimizer_args = mapping("optimizer.args")
    for field in ("name", "exp_name", "dataset.type", "optimizer.type"):
        string(field)
    model_type = string("arch.type")
    if model_type not in ("tinyclip", "openclip", "mobileclip"):
        fail("arch.type", "must be one of tinyclip, openclip, mobileclip")
    for field in ("arch.args.model_arch", "arch.args.pretrained_path"):
        string(field)
    if model_type == "mobileclip":
        string("arch.args.model_config_path")

    for field in (
        "training.batch_size",
        "training.accumulate_grad_batches",
        "training.max_epochs",
        "training.val_check_interval",
        "training.checkpoint_interval_steps",
    ):
        positive_integer(field)
    for field in (
        "loss.distillation_temperature",
        "loss.teacher_soft_label_temperature",
        "optimizer.args.lr",
    ):
        number(field, positive=True)
    warmup_rate = number("scheduler.warmup_ratio", positive=True)
    if warmup_rate >= 1:
        fail("scheduler.warmup_ratio", "must be > 0 and < 1")
    number("scheduler.min_lr")
    if "weight_decay" in optimizer_args:
        number("optimizer.args.weight_decay")
    string_list("arch.trainable_layers")
    loss_config = mapping("loss")
    boolean("loss.enable_cusa")
    enable_hnm = boolean("loss.enable_hnm")
    if enable_hnm or "hnm_threshold" in loss_config:
        number("loss.hnm_threshold")
    if enable_hnm or "hnm_top_k" in loss_config:
        positive_integer("loss.hnm_top_k")

    for split in ("train", "val"):
        image_field = f"dataset.args.{split}_image_root"
        text_field = f"dataset.args.{split}_annotation_files"
        image_roots = string_list(image_field)
        annotations = string_list(text_field)
        if len(image_roots) != len(annotations):
            fail(text_field, f"must have the same length as {image_field}")

    augmentation_field = "dataset.args.train_augmentation"
    mapping(augmentation_field)
    for transform in ("random_resized_crop", "color_jitter", "gaussian_blur"):
        mapping(f"{augmentation_field}.{transform}")

    positive_pair(f"{augmentation_field}.random_resized_crop.scale", maximum=1)
    positive_pair(f"{augmentation_field}.random_resized_crop.ratio")
    positive_pair(f"{augmentation_field}.gaussian_blur.sigma")
    for field in (
        f"{augmentation_field}.horizontal_flip_probability",
        f"{augmentation_field}.gaussian_blur.probability",
    ):
        if number(field) > 1:
            fail(field, "must be between 0 and 1")
    for component in ("brightness", "contrast", "saturation", "hue"):
        field = f"{augmentation_field}.color_jitter.{component}"
        value = number(field)
        if component == "hue" and value > 0.5:
            fail(field, "must be between 0 and 0.5")
    kernel_field = f"{augmentation_field}.gaussian_blur.kernel_size"
    positive_integer(kernel_field)
    if required(kernel_field) % 2 == 0:
        fail(kernel_field, "must be odd")
