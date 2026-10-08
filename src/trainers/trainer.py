import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from src.losses import (
    build_student_distributions,
    build_teacher_distributions,
    compute_cusa_loss,
    compute_hnm_loss,
    compute_isd_loss,
    compute_itc_loss,
    compute_sass_loss,
)
from src.models.build import build_model


class RetrievalTrainer(pl.LightningModule):
    """Lightning module for contrastive CLIP training with distillation losses."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.distillation_criterion = nn.KLDivLoss(reduction="batchmean")
        self.model = build_model(self.config)
        loss_config = self.config["loss"]
        self.distillation_temperature = loss_config["distillation_temperature"]
        self.teacher_soft_label_temperature = loss_config[
            "teacher_soft_label_temperature"
        ]
        self.enable_cusa = loss_config["enable_cusa"]
        self.enable_hnm = loss_config["enable_hnm"]
        self.hnm_threshold = loss_config.get("hnm_threshold")
        self.hnm_top_k = loss_config.get("hnm_top_k")

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Forward inputs to the underlying CLIP model without changing its outputs."""
        return self.model(image, text)

    def _model_logit_scale(self) -> float | torch.Tensor:
        """Return the fixed or learned CLIP logit scale used by the model."""
        if hasattr(self.model, "scale"):
            return self.model.scale
        return self.model._exponentiate_and_clip_logits()

    def _log_training_metrics(self, loss, image_to_text_accuracy, text_to_image_accuracy):
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_image2textacc",
            image_to_text_accuracy,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_text2imageacc",
            text_to_image_accuracy,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "lr",
            current_lr,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, _batch_idx):
        (
            image,
            tokenized_text,
            teacher_text_embeddings,
            teacher_image_embeddings,
        ) = batch
        (
            logits_per_image,
            logits_per_text,
            image_embeddings,
            text_embeddings,
            _projected_image_embeddings,
        ) = self.model(image, tokenized_text)
        itc_loss = compute_itc_loss(
            logits_per_image, logits_per_text, image.shape[0], self.device
        )
        total_loss = itc_loss

        logit_scale = self._model_logit_scale()
        if self.enable_hnm:
            hnm_loss_t2i = compute_hnm_loss(
                text_embeddings,
                image_embeddings,
                logit_scale,
                self.hnm_threshold,
                self.hnm_top_k,
            )
            hnm_loss_i2t = compute_hnm_loss(
                image_embeddings,
                text_embeddings,
                logit_scale,
                self.hnm_threshold,
                self.hnm_top_k,
            )
            hnm_loss = (hnm_loss_t2i + hnm_loss_i2t) / 2
            total_loss = total_loss + hnm_loss

        student_t2i, student_i2t = build_student_distributions(
            logits_per_image,
            logits_per_text,
        )
        teacher_t2i, teacher_i2t, teacher_i2i, teacher_t2t = (
            build_teacher_distributions(
                teacher_text_embeddings,
                teacher_image_embeddings,
                self.teacher_soft_label_temperature,
            )
        )

        sass_loss = compute_sass_loss(
            student_t2i, student_i2t, teacher_t2i, teacher_i2t
        )
        if self.enable_cusa:
            cusa_loss = compute_cusa_loss(
                student_t2i,
                student_i2t,
                teacher_i2i,
                teacher_t2t,
                image_embeddings,
                teacher_image_embeddings,
                logit_scale,
                self.distillation_criterion,
                self.distillation_temperature,
            )
            total_loss = total_loss + cusa_loss

        isd_loss = compute_isd_loss(
            image_embeddings,
            teacher_image_embeddings,
            logit_scale,
            self.distillation_criterion,
            self.distillation_temperature,
        )

        total_loss = total_loss + isd_loss + sass_loss


        image_to_text_accuracy, text_to_image_accuracy = self.batch_acc(
            logits_per_image, logits_per_text
        )
        self._log_training_metrics(
            total_loss, image_to_text_accuracy, text_to_image_accuracy
        )
        return {"loss": total_loss}

    def _reset_validation_retrieval_counts(self):
        """Reset the counters used by validation-epoch retrieval metrics."""
        self.image2text_match = 0
        self.image2text_mismatch = 0
        self.text2image_match = 0
        self.text2image_mismatch = 0

    def on_validation_epoch_start(self):
        self._reset_validation_retrieval_counts()

    def validation_step(self, batch, _batch_idx):
        image, tokenized_text = batch
        (
            logits_per_image,
            logits_per_text,
            _,
            _,
            _projected_image_embeddings,
        ) = self.model(image, tokenized_text)
        # Validation batches have no teacher embeddings, so validation uses ITC only.
        itc_loss = compute_itc_loss(
            logits_per_image, logits_per_text, image.shape[0], self.device
        )
        correct_image_to_text, correct_text_to_image, batch_size = (
            self._batch_retrieval_counts(logits_per_image, logits_per_text)
        )
        self._update_validation_retrieval_counts(
            correct_image_to_text, correct_text_to_image, batch_size
        )
        self.log(
            "val_loss",
            itc_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return {"loss": itc_loss}

    def on_validation_epoch_end(self):
        text2image_acc = self.text2image_match / (
            self.text2image_match + self.text2image_mismatch
        )
        image2text_acc = self.image2text_match / (
            self.image2text_match + self.image2text_mismatch
        )

        self.log(
            "text2image_acc_val",
            text2image_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "image2text_acc_val",
            image2text_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self._reset_validation_retrieval_counts()

    def _batch_retrieval_counts(self, logits_per_image, logits_per_text):
        """Return correct image/text retrieval counts for one in-batch matrix."""
        batch_size = logits_per_image.shape[0]
        row_max_indices = logits_per_image.argmax(dim=1)
        correct_image_to_text = (
            row_max_indices == torch.arange(batch_size).to(logits_per_image.device)
        ).sum()
        row_max_indices = logits_per_text.argmax(dim=1)
        correct_text_to_image = (
            row_max_indices == torch.arange(batch_size).to(logits_per_text.device)
        ).sum()
        return correct_image_to_text, correct_text_to_image, batch_size

    @staticmethod
    def _accuracy_from_counts(correct_image_to_text, correct_text_to_image, batch_size):
        return correct_image_to_text / batch_size, correct_text_to_image / batch_size

    def _update_validation_retrieval_counts(
        self, correct_image_to_text, correct_text_to_image, batch_size
    ):
        """Accumulate counts used by the validation-epoch retrieval metrics."""
        self.image2text_match += correct_image_to_text
        self.image2text_mismatch += batch_size - correct_image_to_text
        self.text2image_match += correct_text_to_image
        self.text2image_mismatch += batch_size - correct_text_to_image

    def batch_acc(self, logits_per_image, logits_per_text):
        """Compute batch-local retrieval accuracy without modifying validation state."""
        counts = self._batch_retrieval_counts(logits_per_image, logits_per_text)
        return self._accuracy_from_counts(*counts)

    def _build_optimizer(self):
        optimizer_settings = self.config["optimizer"]
        optimizer_name = optimizer_settings["type"]
        optimizer_class = getattr(optim, optimizer_name, None)
        if (
            optimizer_class is None
            or not isinstance(optimizer_class, type)
            or not issubclass(optimizer_class, optim.Optimizer)
        ):
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
        return optimizer_class(
            self.parameters(), **optimizer_settings["args"]
        )

    # def on_before_optimizer_step(self, optimizer):
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def configure_optimizers(self):
        optimizer = self._build_optimizer()

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps * self.config["scheduler"]["warmup_ratio"]

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config["scheduler"]["min_lr"],
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        return optimizer_config
