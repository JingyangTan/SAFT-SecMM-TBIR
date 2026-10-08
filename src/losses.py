import torch
from torch.nn import functional as F


def compute_itc_loss(logits_per_image, logits_per_text, batch_size, device) -> torch.Tensor:
    """Compute the bidirectional image-text contrastive (ITC) loss."""
    labels = torch.arange(batch_size).to(device)
    return (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2


def kl_divergence(input_distribution, target_distribution) -> torch.Tensor:
    """Compute ``KL(target || input)`` using the existing epsilon convention."""
    return F.kl_div(
        torch.log(input_distribution + 1e-5),
        target_distribution,
        reduction="batchmean",
        log_target=False,
    )


def build_student_distributions(logits_per_image, logits_per_text) -> tuple[torch.Tensor, ...]:
    """Build the student's cross-modal I2T and T2I distributions."""
    student_t2i = F.softmax(logits_per_text, dim=-1)
    student_i2t = F.softmax(logits_per_image, dim=-1)
    return student_t2i, student_i2t


def build_teacher_distributions(
    teacher_text_embeddings,
    teacher_image_embeddings,
    soft_label_temperature,
) -> tuple[torch.Tensor, ...]:
    """Build teacher cross-modal and intra-modal soft-label distributions."""
    teacher_t2i = F.softmax(
        teacher_text_embeddings
        @ teacher_image_embeddings.t()
        / soft_label_temperature,
        dim=-1,
    )
    teacher_i2t = F.softmax(
        teacher_image_embeddings
        @ teacher_text_embeddings.t()
        / soft_label_temperature,
        dim=-1,
    )
    teacher_i2i = F.softmax(
        teacher_image_embeddings
        @ teacher_image_embeddings.t()
        / soft_label_temperature,
        dim=-1,
    )
    teacher_t2t = F.softmax(
        teacher_text_embeddings
        @ teacher_text_embeddings.t()
        / soft_label_temperature,
        dim=-1,
    )
    return teacher_t2i, teacher_i2t, teacher_i2i, teacher_t2t


def compute_sass_loss(student_t2i, student_i2t, teacher_t2i, teacher_i2t) -> torch.Tensor:
    """Align student and teacher cross-modal distributions with symmetric KL."""
    kl_pq_t2i = kl_divergence(student_t2i, teacher_t2i)
    kl_pq_i2t = kl_divergence(student_i2t, teacher_i2t)
    kl_qp_t2i = kl_divergence(teacher_t2i, student_t2i)
    kl_qp_i2t = kl_divergence(teacher_i2t, student_i2t)
    return (kl_pq_t2i + kl_pq_i2t + kl_qp_t2i + kl_qp_i2t) / 4


def compute_isd_loss(
    student_embeddings,
    teacher_embeddings,
    logit_scale,
    distillation_criterion,
    distillation_temperature,
) -> torch.Tensor:
    """Distill the teacher's intra-modal image structure into the student."""
    logits_s = logit_scale * student_embeddings @ student_embeddings.T
    logits_t = logit_scale * teacher_embeddings @ teacher_embeddings.T
    return distillation_criterion(
        F.log_softmax(logits_s / distillation_temperature, dim=-1),
        F.softmax(logits_t / distillation_temperature, dim=-1),
    ) * distillation_temperature**2


def compute_cusa_loss(
    student_t2i,
    student_i2t,
    teacher_i2i,
    teacher_t2t,
    student_image_embeddings,
    teacher_image_embeddings,
    logit_scale,
    distillation_criterion,
    distillation_temperature,
) -> torch.Tensor:
    """Align cross-modal predictions with teacher intra-modal soft labels.

    The existing objective also includes image-image structural distillation.
    """
    cusa_t2i = kl_divergence(student_t2i, teacher_t2t)
    cusa_i2t = kl_divergence(student_i2t, teacher_i2i)
    cusa_i2i = compute_isd_loss(
        student_image_embeddings,
        teacher_image_embeddings,
        logit_scale,
        distillation_criterion,
        distillation_temperature,
    )
    return 0.5 * (cusa_t2i + cusa_i2t) / 2 + 0.5 * cusa_i2i


def compute_hnm_loss(
    query_embeddings,
    key_embeddings,
    logit_scale,
    threshold,
    top_k,
) -> torch.Tensor:
    """Compute hard-negative-mining InfoNCE loss for one retrieval direction."""
    loss_func = HardNegativeMiningLoss(logit_scale, threshold, top_k)
    return loss_func(query_embeddings, key_embeddings)


class HardNegativeMiningLoss:
    """InfoNCE over positive pairs and negatives selected using a threshold mask."""

    def __init__(self, logit_scale, threshold: float = 0.0, top_k: int = 32):
        self.logit_scale = logit_scale
        self.threshold = threshold
        self.top_k = top_k

    def _select_rows_by_indices(self, matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return matrix[indices]

    def _hard_negative_indices(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Mask the positive pair and candidates above the positive similarity
        plus the threshold before selecting top-k indices.

        The mask applies only to index selection. Final scores are recomputed
        without it; masked entries can participate in the loss when fewer
        than top-k valid candidates remain.

        Args:
            similarity_matrix (torch.Tensor): A square matrix of shape (N, N).
                Entry [i, j] is the similarity between query i and key j.
            The number of top-ranked negatives is set by ``self.top_k``.
            ``self.threshold`` is added to each positive similarity to set the
            false-negative filtering threshold.

        Returns:
            torch.Tensor: Indices of selected hard negatives, with shape
                ``(N, self.top_k)``.
        """
        n = similarity_matrix.size(0)
        # Clone the matrix to avoid modifying the original similarity_matrix
        modified_matrix = similarity_matrix.clone()
        # Get the diagonal values (positive similarities)
        diag_values = torch.diag(similarity_matrix)

        # Exclude off-diagonal candidates above the positive-based threshold
        # from index selection.
        threshold = diag_values.view(-1, 1) + self.threshold
        threshold_mask = (modified_matrix > threshold) & (
            ~torch.eye(n, dtype=torch.bool, device=modified_matrix.device)
        )

        modified_matrix[threshold_mask] = -float('inf')
        modified_matrix.fill_diagonal_(-float('inf'))

        _, topk_indices = torch.topk(modified_matrix, self.top_k, dim=1)
        return topk_indices

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        similarity_matrix = query_embeddings @ key_embeddings.T
        neg_indices = self._hard_negative_indices(similarity_matrix)
        neg_keys = self._select_rows_by_indices(key_embeddings, neg_indices)

        pos_key = key_embeddings.unsqueeze(1)
        all_keys = torch.cat([pos_key, neg_keys], dim=1)
        query = query_embeddings.unsqueeze(1)
        scores = self.logit_scale * torch.bmm(query, all_keys.transpose(1, 2)).squeeze(1)
        labels = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
        return F.cross_entropy(scores, labels, reduction=reduction)
