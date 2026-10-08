import open_clip
import torch
import torch.nn as nn
from torch.nn import functional as F


class OpenCLIPWrapper(nn.Module):
    """OpenCLIP wrapper used by the training pipeline."""

    def __init__(self, config):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            config["arch"]["args"]["model_arch"],
            pretrained=config["arch"]["args"]["pretrained_path"],
        )
        self.scale = 100.0

        student_dim = self.model.visual.proj.shape[1]
        self.image_proj = nn.Linear(student_dim, student_dim, bias=False)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Encode a paired image-text batch and return five tensors in order.

        Both image and text must be supplied for this forward path.

        Args:
            image: Preprocessed RGB images with shape [B, 3, 224, 224].
                The current pipeline applies CLIP normalization.
            text: Token IDs with shape [B, L], produced by the OpenCLIP tokenizer.

        Returns:
            A tuple containing:
                1. logits_per_image: I2T logits, shape [B, B].
                2. logits_per_text: T2I logits, shape [B, B].
                3. image_embeddings: L2-normalized image features, shape [B, D].
                4. text_embeddings: L2-normalized text features, shape [B, D].
                5. projected_image_embeddings: L2-normalized projected image
                   features, shape [B, D].
            B is batch size, L is token sequence length, and D is the model's
            embedding dimension. The projection uses image features before external normalization.
            The current trainer does not use the projected output in its losses.
        """

        image_embeddings = self.model.encode_image(image)
        text_embeddings = self.model.encode_text(text)

        # Keep the projected image embeddings in the shared model interface.
        projected_image_embeddings = self.image_proj(image_embeddings)
        projected_image_embeddings = F.normalize(projected_image_embeddings, dim=-1)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        logits_per_image = self.scale * image_embeddings @ text_embeddings.t()
        logits_per_text = self.scale * text_embeddings @ image_embeddings.t()
        return (
            logits_per_image,
            logits_per_text,
            image_embeddings,
            text_embeddings,
            projected_image_embeddings,
        )


