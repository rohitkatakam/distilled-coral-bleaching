"""Tests for knowledge distillation utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from models.distillation import (
    DistillationLoss,
    EPSILON,
    kl_divergence,
    temperature_scaled_softmax,
)


class TestTemperatureScaledSoftmax:
    """Validate the manual softmax with temperature scaling."""

    @pytest.mark.parametrize("temperature", [1.0, 2.5, 5.0])
    @pytest.mark.parametrize("dim", [-1, 1])
    def test_matches_torch_softmax(self, temperature, dim):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 4)
        expected = torch.softmax(logits / temperature, dim=dim)
        actual = temperature_scaled_softmax(logits, temperature, dim=dim)
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-5)

    def test_handles_large_magnitude_logits(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1500.0, 1500.0]])
        probs = temperature_scaled_softmax(logits, temperature=1.0, dim=-1)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        assert torch.isfinite(probs).all()

    def test_invalid_temperature_raises(self):
        logits = torch.randn(2, 3)
        with pytest.raises(ValueError):
            temperature_scaled_softmax(logits, temperature=0.0)


class TestKLDivergence:
    """Validate manual KL divergence implementation."""

    @pytest.fixture
    def probability_pair(self):
        torch.manual_seed(42)
        teacher_logits = torch.randn(5, 4)
        student_logits = torch.randn(5, 4)
        return torch.softmax(teacher_logits, dim=-1), torch.softmax(student_logits, dim=-1)

    def test_matches_manual_computation(self, probability_pair):
        p_teacher, p_student = probability_pair
        manual = kl_divergence(p_teacher, p_student, reduction="batchmean")

        # Reproduce calculation explicitly for clarity.
        p_t = p_teacher.clamp_min(EPSILON)
        p_s = p_student.clamp_min(EPSILON)
        log_ratio = torch.log(p_t) - torch.log(p_s)
        expected = (p_t * log_ratio).sum(dim=-1).mean()

        assert torch.allclose(manual, expected, atol=1e-7, rtol=1e-5)

    def test_matches_torch_functional(self, probability_pair):
        p_teacher, p_student = probability_pair
        manual = kl_divergence(p_teacher, p_student, reduction="batchmean")
        torch_equiv = F.kl_div(
            torch.log(p_student.clamp_min(EPSILON)),
            p_teacher.clamp_min(EPSILON),
            reduction="batchmean",
        )
        assert torch.allclose(manual, torch_equiv, atol=1e-7, rtol=1e-5)

    def test_numerical_stability_near_zero(self):
        p_teacher = torch.tensor([[1.0 - 1e-7, 1e-7]])
        p_student = torch.tensor([[0.5, 0.5]])
        value = kl_divergence(p_teacher, p_student, reduction="batchmean")
        assert torch.isfinite(value)
        assert value > 0


class TestDistillationLoss:
    """End-to-end tests for the composite loss."""

    def test_total_loss_matches_formula(self):
        temperature = 2.0
        alpha = 0.3
        criterion = DistillationLoss(temperature=temperature, alpha=alpha)

        student_logits = torch.tensor([[2.0, 0.5]], requires_grad=True)
        teacher_logits = torch.tensor([[1.5, 0.2]], requires_grad=True)
        hard_labels = torch.tensor([0])

        total_loss, metrics = criterion(student_logits, teacher_logits, hard_labels)

        teacher_probs = temperature_scaled_softmax(teacher_logits.detach(), temperature)
        student_probs = temperature_scaled_softmax(student_logits, temperature)
        kd = (temperature ** 2) * kl_divergence(teacher_probs, student_probs)
        ce = nn.CrossEntropyLoss()(student_logits, hard_labels)
        expected_total = alpha * kd + (1 - alpha) * ce

        assert torch.allclose(total_loss, expected_total, atol=1e-7, rtol=1e-5)
        assert torch.allclose(metrics["kd_loss"], kd.detach())
        assert torch.allclose(metrics["hard_loss"], ce.detach())

    def test_alpha_zero_reduces_to_cross_entropy(self):
        criterion = DistillationLoss(temperature=2.0, alpha=0.0)
        student_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        teacher_logits = torch.tensor([[0.5, 0.5]], requires_grad=True)
        labels = torch.tensor([1])

        total_loss, metrics = criterion(student_logits, teacher_logits, labels)
        ce = nn.CrossEntropyLoss()(student_logits, labels)

        assert torch.allclose(total_loss, ce, atol=1e-7)
        assert torch.allclose(metrics["hard_loss"], ce.detach(), atol=1e-7)

    def test_alpha_one_without_hard_loss(self):
        criterion = DistillationLoss(temperature=3.0, alpha=1.0)
        student_logits = torch.tensor([[0.2, 1.5]], requires_grad=True)
        teacher_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])

        total_loss, metrics = criterion(
            student_logits, teacher_logits, labels, include_hard_loss=False
        )

        teacher_probs = temperature_scaled_softmax(teacher_logits.detach(), 3.0)
        student_probs = temperature_scaled_softmax(student_logits, 3.0)
        kd = (3.0 ** 2) * kl_divergence(teacher_probs, student_probs)

        assert torch.allclose(total_loss, kd, atol=1e-7)
        assert torch.allclose(metrics["kd_loss"], kd.detach(), atol=1e-7)
        assert torch.allclose(metrics["hard_loss"], torch.tensor(0.0), atol=1e-7)

    def test_backward_only_updates_student(self):
        criterion = DistillationLoss(temperature=2.0, alpha=0.5)
        student_logits = torch.tensor([[0.3, 1.2]], requires_grad=True)
        teacher_logits = torch.tensor([[1.0, -0.5]], requires_grad=True)
        labels = torch.tensor([1])

        total_loss, _ = criterion(student_logits, teacher_logits, labels)
        total_loss.backward()

        assert student_logits.grad is not None
        assert torch.all(student_logits.grad.abs() > 0)
        assert teacher_logits.grad is None  # teacher path should be detached

    def test_invalid_temperature_in_constructor(self):
        with pytest.raises(ValueError):
            DistillationLoss(temperature=0.0, alpha=0.5)
