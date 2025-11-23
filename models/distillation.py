"""Knowledge distillation utilities for coral bleaching classification.

The key idea in distillation is to match the student distribution
``p_student`` to the teacher distribution ``p_teacher`` after both are
softened with a *temperature* ``T``. Dividing logits by ``T`` produces
``p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)``, which amplifies secondary
classes for ``T > 1``. Those richer targets carry information about the
teacher's belief over *all* classes rather than just the arg-max label.

Because ``softmax(z / T)`` introduces a ``1 / T`` factor in the gradient
``∂ log softmax(z / T) / ∂ z``, the gradient magnitude would shrink as
``T`` grows. Hinton et al. (2015) multiply the KL term by ``T^2`` so that
``∂ (T^2 * KL) / ∂ z`` roughly matches the scale of the original
cross-entropy gradients, keeping the optimization dynamics comparable.
The implementations below explicitly carry out each numerical step so
that the math is transparent and easily auditable for coursework.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

EPSILON: float = 1e-8

__all__ = ["EPSILON", "temperature_scaled_softmax", "kl_divergence", "DistillationLoss"]


def _validate_tensor(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating point tensor, got dtype={tensor.dtype}")


def temperature_scaled_softmax(logits: Tensor, temperature: float, dim: int = -1) -> Tensor:
    """Compute ``softmax(logits / T)`` manually with stability safeguards.

    Args:
        logits: Raw model outputs. Can have arbitrary rank as long as ``dim`` is valid.
        temperature: Positive scaling factor ``T``. Values ``T > 1`` soften predictions.
        dim: Dimension representing the class axis.

    Returns:
        Tensor of probabilities that sums to 1 along ``dim``.

    Notes:
        We subtract the maximum value prior to exponentiation to avoid
        overflow, mirroring the usual log-sum-exp trick.
    """

    _validate_tensor("logits", logits)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    scaled = logits / temperature
    max_vals, _ = torch.max(scaled, dim=dim, keepdim=True)
    shifted = scaled - max_vals  # improves numerical stability
    exp_shifted = shifted.exp()
    partition = exp_shifted.sum(dim=dim, keepdim=True).clamp_min(EPSILON)
    return exp_shifted / partition


def kl_divergence(
    p_teacher: Tensor,
    p_student: Tensor,
    reduction: str = "batchmean",
    eps: float = EPSILON,
) -> Tensor:
    """Compute KL(p_teacher || p_student) in probability space.

    Args:
        p_teacher: Teacher probabilities (already softmaxed).
        p_student: Student probabilities (already softmaxed).
        reduction: ``"batchmean"``, ``"sum"``, or ``"none"``.
        eps: Minimum probability used when taking logs to avoid ``log(0)``.

    Returns:
        Scalar tensor when reduction is ``batchmean`` or ``sum``; otherwise
        the per-example KL values.

    This mirrors ``torch.nn.functional.kl_div`` when feeding it log
    probabilities, but we stay in probability space to satisfy the
    course requirement of demonstrating the underlying arithmetic.
    """

    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    _validate_tensor("p_teacher", p_teacher)
    _validate_tensor("p_student", p_student)
    if p_teacher.shape != p_student.shape:
        raise ValueError(
            "p_teacher and p_student must have identical shapes for KL computation."
        )

    # Light-range check keeps debugging easy without being too strict for float noise.
    if torch.any(p_teacher < -1e-6) or torch.any(p_teacher > 1.0 + 1e-6):
        raise ValueError("p_teacher must contain probabilities in [0, 1].")
    if torch.any(p_student < -1e-6) or torch.any(p_student > 1.0 + 1e-6):
        raise ValueError("p_student must contain probabilities in [0, 1].")

    p_teacher = p_teacher.clamp(min=eps, max=1.0)
    p_student = p_student.clamp(min=eps, max=1.0)

    log_ratio = torch.log(p_teacher) - torch.log(p_student)
    kl_per_class = p_teacher * log_ratio
    per_example = kl_per_class.sum(dim=-1)

    if reduction not in {"batchmean", "sum", "none"}:
        raise ValueError(f"Unsupported reduction '{reduction}'.")

    if reduction == "batchmean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    return per_example


class DistillationLoss(nn.Module):
    """Blend temperature-scaled KL loss with hard-label cross-entropy."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        hard_weight: Optional[float] = None,
        reduction: str = "batchmean",
        eps: float = EPSILON,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1].")

        if reduction not in {"batchmean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction '{reduction}'.")
        self.reduction = reduction
        self.eps = float(eps)

        if hard_weight is None:
            self.hard_weight = 1.0 - self.alpha
        else:
            self.hard_weight = float(hard_weight)
            if self.hard_weight < 0.0:
                raise ValueError("hard_weight must be non-negative.")
            expected = 1.0 - self.alpha
            if self.hard_weight > 0.0 and not math.isclose(
                self.hard_weight, expected, rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError(
                    "hard_weight should match (1 - alpha) unless you explicitly "
                    "disable the hard loss."
                )

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        hard_labels: Optional[Tensor],
        include_hard_loss: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the combined distillation and hard losses.

        Args:
            student_logits: Student raw outputs.
            teacher_logits: Teacher raw outputs (will be detached).
            hard_labels: Integer class labels for the hard CE term.
            include_hard_loss: If False, skips CE yet still returns a zeroed
                ``hard_loss`` entry so logging code can remain uniform.

        Returns:
            Tuple of ``(total_loss, metrics_dict)`` where metrics contain both
            components for logging.
        """

        _validate_tensor("student_logits", student_logits)
        _validate_tensor("teacher_logits", teacher_logits)
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("student_logits and teacher_logits must share the same shape.")

        if include_hard_loss and hard_labels is None:
            raise ValueError("hard_labels must be provided when include_hard_loss=True.")

        teacher_detached = teacher_logits.detach()
        teacher_probs = temperature_scaled_softmax(
            teacher_detached, self.temperature, dim=-1
        )
        student_probs = temperature_scaled_softmax(student_logits, self.temperature, dim=-1)

        # Multiply by T^2 so that gradients wrt logits stay comparable after the 1/T scaling.
        kd_loss = (self.temperature ** 2) * kl_divergence(
            teacher_probs, student_probs, reduction=self.reduction, eps=self.eps
        )

        hard_loss = student_logits.new_tensor(0.0)
        if include_hard_loss and self.hard_weight > 0.0:
            hard_loss = self.cross_entropy(student_logits, hard_labels)

        hard_coeff = self.hard_weight if include_hard_loss else 0.0
        total_loss = self.alpha * kd_loss + hard_coeff * hard_loss

        metrics = {
            "kd_loss": kd_loss.detach(),
            "hard_loss": hard_loss.detach(),
            "temperature": torch.tensor(self.temperature, device=student_logits.device),
            "alpha": torch.tensor(self.alpha, device=student_logits.device),
        }
        return total_loss, metrics
