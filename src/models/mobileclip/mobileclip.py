import math
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.mobileclip.mobileclip_image_encoder import MCi
from src.models.mobileclip.mobileclip_text_encoder import TextTransformer


class MobileCLIPWrapper(nn.Module):
    """MobileCLIP wrapper used by the training pipeline."""

    def __init__(
        self,
        mobileclip_cfg: Dict,
    ) -> None:
        super().__init__()
        self.projection_dim = mobileclip_cfg["embed_dim"]
        if self.projection_dim is None:
            raise ValueError("Please specify `embed_dim` in model config.")

        self.image_encoder = MCi(
            model_name=mobileclip_cfg["image_cfg"]["model_name"],
            projection_dim=self.projection_dim,
        )
        self.text_encoder = TextTransformer(
            mobileclip_cfg=mobileclip_cfg["text_cfg"], projection_dim=self.projection_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        student_dim = self.image_encoder.model.head.proj.shape[1]
        self.image_proj = nn.Linear(student_dim, student_dim, bias=False)

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0) -> torch.Tensor:
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        image_encoder_out = self.image_encoder(image)
        if isinstance(image_encoder_out, dict):
            features = image_encoder_out["logits"]
        else:
            features = image_encoder_out
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Encode a paired image-text batch and return five tensors in order.

        Both image and text must be supplied for this forward path.

        Args:
            image: Preprocessed RGB images with shape [B, 3, 256, 256].
                The current pipeline does not apply external CLIP normalization.
            text: Token IDs with shape [B, L], produced by the MobileCLIP tokenizer.

        Returns:
            A tuple containing:
                1. logits_per_image: I2T logits, shape [B, B].
                2. logits_per_text: T2I logits, shape [B, B].
                3. image_embeddings: L2-normalized image features, shape [B, D].
                4. text_embeddings: L2-normalized text features, shape [B, D].
                5. projected_image_embeddings: L2-normalized projected image
                   features, shape [B, D].
            B is batch size, L is token sequence length, and D is the model's
            embedding dimension. The projection uses normalized image features.
            The current trainer does not use the projected output in its losses.
        """
        image_embeddings = self.encode_image(image, normalize=True)
        text_embeddings = self.encode_text(text, normalize=True)

        projected_image_embeddings = self.image_proj(image_embeddings)
        projected_image_embeddings = F.normalize(projected_image_embeddings, p=2, dim=-1)

        logit_scale = self._exponentiate_and_clip_logits()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.t()
        return (
            logits_per_image,
            logits_per_text,
            image_embeddings,
            text_embeddings,
            projected_image_embeddings,
        )
