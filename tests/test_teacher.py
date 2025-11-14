"""Tests for teacher model implementation."""

import pytest
import torch
import torch.nn as nn

from models.teacher import TeacherModel


class TestTeacherModelInitialization:
    """Test teacher model initialization with various configurations."""

    def test_default_initialization(self):
        """Test model initializes with default parameters."""
        # Use pretrained=False to avoid downloading weights during testing
        model = TeacherModel(pretrained=False)
        assert isinstance(model, nn.Module)
        assert model.num_classes == 2
        assert model.dropout_p is None

    def test_custom_num_classes(self):
        """Test model initializes with custom number of classes."""
        model = TeacherModel(num_classes=5, pretrained=False)
        assert model.num_classes == 5

        # Check that final FC layer has correct output size
        # The final layer might be wrapped in Sequential if dropout is present
        if isinstance(model.backbone.fc, nn.Sequential):
            fc_layer = model.backbone.fc[-1]
        else:
            fc_layer = model.backbone.fc
        assert fc_layer.out_features == 5

    @pytest.mark.skip(reason="Requires network access to download pretrained weights")
    def test_pretrained_weights_loading(self):
        """Test that pretrained weights are loaded."""
        model_pretrained = TeacherModel(pretrained=True)
        model_random = TeacherModel(pretrained=False)

        # Get weights from first conv layer
        pretrained_weights = model_pretrained.backbone.conv1.weight.data.clone()
        random_weights = model_random.backbone.conv1.weight.data.clone()

        # Pretrained and random weights should be different
        # (extremely unlikely to be the same)
        assert not torch.allclose(pretrained_weights, random_weights)

    def test_no_pretrained_weights(self):
        """Test model initializes without pretrained weights."""
        model = TeacherModel(pretrained=False)
        assert model.pretrained is False
        # Model should still be functional
        assert isinstance(model, nn.Module)

    def test_with_dropout(self):
        """Test model initializes with dropout."""
        dropout_p = 0.5
        model = TeacherModel(dropout=dropout_p, pretrained=False)
        assert model.dropout_p == dropout_p

        # Check that FC layer is wrapped in Sequential with Dropout
        assert isinstance(model.backbone.fc, nn.Sequential)
        assert isinstance(model.backbone.fc[0], nn.Dropout)
        assert model.backbone.fc[0].p == dropout_p
        assert isinstance(model.backbone.fc[1], nn.Linear)

    def test_without_dropout(self):
        """Test model initializes without dropout when None."""
        model = TeacherModel(dropout=None, pretrained=False)
        assert model.dropout_p is None

        # Check that FC layer is a simple Linear layer
        assert isinstance(model.backbone.fc, nn.Linear)

    def test_zero_dropout(self):
        """Test model initializes without dropout when dropout=0."""
        model = TeacherModel(dropout=0.0, pretrained=False)
        assert model.dropout_p == 0.0

        # Should not add dropout layer for zero dropout
        assert isinstance(model.backbone.fc, nn.Linear)


class TestTeacherModelForward:
    """Test teacher model forward pass."""

    @pytest.fixture
    def model(self):
        """Create a teacher model for testing."""
        return TeacherModel(pretrained=False)  # Use random weights for speed

    def test_forward_pass_shape(self, model):
        """Test forward pass returns correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x)

        assert output.shape == (batch_size, 2)

    def test_forward_pass_different_batch_sizes(self, model):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_forward_pass_no_nan(self, model):
        """Test forward pass output contains no NaN values."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert not torch.isnan(output).any()

    def test_forward_pass_no_inf(self, model):
        """Test forward pass output contains no infinite values."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert not torch.isinf(output).any()

    def test_forward_pass_custom_num_classes(self):
        """Test forward pass with custom number of classes."""
        num_classes = 5
        model = TeacherModel(num_classes=num_classes, pretrained=False)
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert output.shape == (2, num_classes)

    def test_forward_pass_with_dropout(self):
        """Test forward pass works with dropout enabled."""
        model = TeacherModel(dropout=0.5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)

        # Test in training mode (dropout active)
        model.train()
        output_train = model(x)
        assert output_train.shape == (2, 2)

        # Test in eval mode (dropout inactive)
        model.eval()
        output_eval = model(x)
        assert output_eval.shape == (2, 2)

    def test_backward_pass(self, model):
        """Test that gradients can be computed (model is trainable)."""
        x = torch.randn(2, 3, 224, 224)
        target = torch.tensor([0, 1])

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed during backward pass"


class TestTeacherModelFreezing:
    """Test backbone freezing and unfreezing functionality."""

    @pytest.fixture
    def model(self):
        """Create a teacher model for testing."""
        return TeacherModel(pretrained=False)

    def test_initial_all_trainable(self, model):
        """Test that all parameters are trainable initially."""
        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_backbone(self, model):
        """Test freezing backbone layers."""
        model.freeze_backbone()

        # Check that backbone conv layers are frozen
        for name, param in model.backbone.named_parameters():
            if 'fc' not in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"

        # Check that final FC layer is still trainable
        for param in model.backbone.fc.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model):
        """Test unfreezing backbone layers."""
        # First freeze
        model.freeze_backbone()

        # Then unfreeze
        model.unfreeze_backbone()

        # Check that all parameters are trainable again
        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_unfreeze_cycle(self, model):
        """Test multiple freeze/unfreeze cycles."""
        # Initial state: all trainable
        assert all(p.requires_grad for p in model.parameters())

        # Freeze
        model.freeze_backbone()
        # Only FC should be trainable
        fc_params = list(model.backbone.fc.parameters())
        assert all(p.requires_grad for p in fc_params)

        # Unfreeze
        model.unfreeze_backbone()
        # All should be trainable
        assert all(p.requires_grad for p in model.parameters())

        # Freeze again
        model.freeze_backbone()
        # Only FC should be trainable
        assert all(p.requires_grad for p in fc_params)

    def test_gradient_computation_when_frozen(self, model):
        """Test that only FC layer receives gradients when backbone is frozen."""
        model.freeze_backbone()

        x = torch.randn(2, 3, 224, 224)
        target = torch.tensor([0, 1])

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check that backbone conv layers have no gradients
        for name, param in model.backbone.named_parameters():
            if 'fc' not in name:
                assert param.grad is None, f"Parameter {name} should have no gradient"

        # Check that FC layer has gradients
        for param in model.backbone.fc.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0


class TestTeacherModelParameterCounting:
    """Test parameter counting methods."""

    @pytest.fixture
    def model(self):
        """Create a teacher model for testing."""
        return TeacherModel(pretrained=False)

    def test_get_num_total_params(self, model):
        """Test total parameter counting."""
        total_params = model.get_num_total_params()

        # ResNet50 has approximately 25.6M parameters
        # With 2-class FC layer, should be around 23-25M parameters
        assert 23_000_000 < total_params < 26_000_000

    def test_get_num_trainable_params_all_unfrozen(self, model):
        """Test trainable parameter counting when all layers are trainable."""
        total_params = model.get_num_total_params()
        trainable_params = model.get_num_trainable_params()

        # Initially, all parameters should be trainable
        assert trainable_params == total_params

    def test_get_num_trainable_params_frozen_backbone(self, model):
        """Test trainable parameter counting when backbone is frozen."""
        model.freeze_backbone()

        trainable_params = model.get_num_trainable_params()

        # Only FC layer should be trainable (2048 * 2 + 2 = 4098 params)
        # Allow some tolerance for different configurations
        assert 4000 < trainable_params < 5000

    def test_trainable_params_change_after_freeze(self, model):
        """Test that trainable param count changes after freezing."""
        trainable_before = model.get_num_trainable_params()

        model.freeze_backbone()

        trainable_after = model.get_num_trainable_params()

        assert trainable_after < trainable_before
        # Most parameters should be frozen
        assert trainable_after < 0.01 * trainable_before

    def test_total_params_unchanged_by_freezing(self, model):
        """Test that total parameter count doesn't change when freezing."""
        total_before = model.get_num_total_params()

        model.freeze_backbone()

        total_after = model.get_num_total_params()

        assert total_before == total_after


class TestTeacherModelIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_training_step(self):
        """Test a complete training step with optimizer."""
        model = TeacherModel(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training step
        model.train()
        x = torch.randn(4, 3, 224, 224)
        target = torch.tensor([0, 1, 0, 1])

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Verify loss is a valid number
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_inference_mode(self):
        """Test model in inference mode."""
        model = TeacherModel(pretrained=False)
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Check output properties
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()

        # Check that we can get predictions
        predictions = output.argmax(dim=1)
        assert predictions.shape == (4,)
        assert all(0 <= p < 2 for p in predictions)

    @pytest.mark.skip(reason="Requires network access to download pretrained weights")
    def test_pretrained_vs_random_performance(self):
        """Test that pretrained and random models produce different outputs."""
        model_pretrained = TeacherModel(pretrained=True)
        model_random = TeacherModel(pretrained=False)

        # Use same random seed for input
        torch.manual_seed(42)
        x = torch.randn(2, 3, 224, 224)

        model_pretrained.eval()
        model_random.eval()

        with torch.no_grad():
            output_pretrained = model_pretrained(x)
            output_random = model_random(x)

        # Outputs should be different (different initialization)
        assert not torch.allclose(output_pretrained, output_random)
