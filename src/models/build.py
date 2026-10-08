import json

import torch

from src.models.mobileclip.mobileclip import MobileCLIPWrapper
from src.models.TinyCLIP.tinyclip import TinyCLIPWrapper
from src.models.clip.clip import OpenCLIPWrapper


def build_mobileclip(config):
    """Build MobileCLIP and load its checkpoint with the existing non-strict policy."""
    with open(config['arch']['args']['model_config_path'], "r") as config_file:
        mobileclip_cfg = json.load(config_file)
    model = MobileCLIPWrapper(mobileclip_cfg=mobileclip_cfg)
    checkpoint = torch.load(config['arch']['args']['pretrained_path'])
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    return model


def build_tinyclip(config):
    """Build TinyCLIP through its model-specific wrapper."""
    return TinyCLIPWrapper(config)


def build_clip(config):
    """Build OpenCLIP through its model-specific wrapper."""
    return OpenCLIPWrapper(config)


def set_trainable_parameters(model, trainable_layers):
    """Set parameter trainability using the existing substring matching rule."""
    parameter_names = [name for name, _ in model.named_parameters()]
    unmatched_layers = [
        layer
        for layer in trainable_layers
        if not any(layer in name for name in parameter_names)
    ]
    if unmatched_layers:
        raise ValueError(f"Unmatched trainable_layers entries: {unmatched_layers}")

    trainable_parameter_count = 0
    frozen_parameter_count = 0
    for name, param in model.named_parameters():
        if any(layer in name for layer in trainable_layers):
            param.requires_grad_(True)
            trainable_parameter_count += param.numel()
            print(name)
        else:
            param.requires_grad_(False)
            frozen_parameter_count += param.numel()

    print(
        f"learnable: {trainable_parameter_count} --- unlearnable: {frozen_parameter_count}"
    )


def build_model(config):
    """Build the selected model and apply its configured trainable layers."""
    model_type = config['arch']['type']
    if model_type == "mobileclip":
        model = build_mobileclip(config)
    elif model_type == "tinyclip":
        model = build_tinyclip(config)
    elif model_type == "openclip":
        model = build_clip(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    set_trainable_parameters(model, config["arch"]["trainable_layers"])
    return model
