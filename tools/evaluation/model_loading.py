import json
import os
import sys

import torch

# openclip,tinyclip
def _get_clip_library(model_type):
    if model_type == "tinyclip":
        repository_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if repository_root not in sys.path:
            sys.path.append(repository_root)
        from src.models.TinyCLIP.src import open_clip as clip_library
    elif model_type == "openclip":
        import open_clip as clip_library
    else:
        raise ValueError(f"Unsupported CLIP model type: {model_type}")
    return clip_library


def initialize_clip_foundation_model(model_type, model_arch, pretrained_path):
    """Initialize TinyCLIP or OpenCLIP through their shared factory interface."""
    clip_library = _get_clip_library(model_type)
    model, _, preprocess = clip_library.create_model_and_transforms(
        model_arch,
        pretrained=pretrained_path,
    )
    tokenizer = clip_library.get_tokenizer(model_arch)
    return model, preprocess, tokenizer

# mobileclip
def build_mobileclip_model(model_arch, model_config_path):
    import mobileclip
    from mobileclip.clip import CLIP
    from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

    with open(model_config_path, encoding="utf-8") as stream:
        model_cfg = json.load(stream)
    model = CLIP(cfg=model_cfg)
    resolution = model_cfg["image_cfg"]["image_size"]
    preprocess = Compose(
        [
            Resize(
                (resolution, resolution),
                interpolation=InterpolationMode.BILINEAR,
            ),
            ToTensor(),
        ]
    )
    tokenizer = mobileclip.get_tokenizer(model_arch)
    return model, preprocess, tokenizer


def _initialize_mobileclip_from_state_dict(model_config, state_dict):
    from mobileclip.modules.common.mobileone import reparameterize_model

    model, preprocess, tokenizer = build_mobileclip_model(
        model_config["architecture"],
        model_config["model_config_path"],
    )
    model.load_state_dict(state_dict, strict=True)
    model = reparameterize_model(model)
    return model, preprocess, tokenizer


def _initialize_mobileclip_foundation_model(model_config):
    state_dict = torch.load(
        model_config["pretrained_path"],
        map_location="cpu",
        weights_only=False,
    )
    return _initialize_mobileclip_from_state_dict(model_config, state_dict)


def initialize_inference_model(model_config, inference_weights_path):
    """Build an inference model and strictly load an exported state dict."""
    state_dict = torch.load(
        inference_weights_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"Inference weights must be a non-empty state dict: "
            f"{inference_weights_path}"
        )

    if model_config["type"] == "mobileclip":
        return _initialize_mobileclip_from_state_dict(model_config, state_dict)

    model, preprocess, tokenizer = initialize_clip_foundation_model(
        model_config["type"],
        model_config["architecture"],
        model_config["pretrained_path"],
    )
    model.load_state_dict(state_dict, strict=True)
    return model, preprocess, tokenizer


def initialize_foundation_model(model_config):
    """Initialize a model from its configured pretrained weights."""
    model_type = model_config["type"]
    if model_type == "mobileclip":
        return _initialize_mobileclip_foundation_model(model_config)
    return initialize_clip_foundation_model(
        model_type,
        model_config["architecture"],
        model_config["pretrained_path"],
    )
