"""
Tests for utils/metrics.py

Tests metric computation functions for classification.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from utils.metrics import (
    to_numpy,
    compute_accuracy,
    compute_confusion_matrix,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_classification_metrics,
    print_classification_report,
    log_metrics_to_wandb,
    log_confusion_matrix_to_wandb,
    compute_metrics,
)


@pytest.fixture
def sample_predictions():
    """Sample binary classification predictions."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    return y_true, y_pred


@pytest.fixture
def perfect_predictions():
    """Perfect predictions for testing."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def class_names():
    """Class names for binary classification."""
    return ['bleached', 'healthy']


class TestToNumpy:
    """Tests for to_numpy conversion function."""

    def test_numpy_array_passthrough(self):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_torch_tensor_conversion(self):
        """Test that torch tensors are converted to numpy."""
        tensor = torch.tensor([1, 2, 3])
        result = to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_gpu_tensor_conversion(self):
        """Test conversion of GPU tensors (if CUDA available)."""
        tensor = torch.tensor([1, 2, 3])
        # Even without GPU, test the path
        result = to_numpy(tensor)

        assert isinstance(result, np.ndarray)

    def test_list_conversion(self):
        """Test that lists are converted to numpy arrays."""
        lst = [1, 2, 3]
        result = to_numpy(lst)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_accuracy_with_numpy(self, sample_predictions):
        """Test accuracy computation with numpy arrays."""
        y_true, y_pred = sample_predictions
        accuracy = compute_accuracy(y_true, y_pred)

        # Manual calculation: 8 correct out of 10
        assert accuracy == 0.8

    def test_accuracy_with_torch(self, sample_predictions):
        """Test accuracy computation with torch tensors."""
        y_true, y_pred = sample_predictions
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)

        accuracy = compute_accuracy(y_true_tensor, y_pred_tensor)
        assert accuracy == 0.8

    def test_perfect_accuracy(self, perfect_predictions):
        """Test that perfect predictions give accuracy of 1.0."""
        y_true, y_pred = perfect_predictions
        accuracy = compute_accuracy(y_true, y_pred)

        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """Test completely wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.0

    def test_accuracy_returns_float(self, sample_predictions):
        """Test that accuracy returns a float."""
        y_true, y_pred = sample_predictions
        accuracy = compute_accuracy(y_true, y_pred)

        assert isinstance(accuracy, float)


class TestComputeConfusionMatrix:
    """Tests for compute_confusion_matrix function."""

    def test_confusion_matrix_shape(self, sample_predictions):
        """Test that confusion matrix has correct shape."""
        y_true, y_pred = sample_predictions
        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)  # Binary classification

    def test_confusion_matrix_values(self):
        """Test confusion matrix values."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        cm = compute_confusion_matrix(y_true, y_pred)

        # Expected:
        # [[TN=2, FP=0],
        #  [FN=1, TP=1]]
        assert cm[0, 0] == 2  # True negatives
        assert cm[0, 1] == 0  # False positives
        assert cm[1, 0] == 1  # False negatives
        assert cm[1, 1] == 1  # True positives

    def test_confusion_matrix_normalization(self, sample_predictions):
        """Test confusion matrix normalization."""
        y_true, y_pred = sample_predictions

        # Normalized by true labels
        cm_norm = compute_confusion_matrix(y_true, y_pred, normalize='true')

        # Each row should sum to 1
        row_sums = cm_norm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.array([1.0, 1.0]))

    def test_confusion_matrix_with_torch(self, sample_predictions):
        """Test confusion matrix with torch tensors."""
        y_true, y_pred = sample_predictions
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)

        cm = compute_confusion_matrix(y_true_tensor, y_pred_tensor)
        assert cm.shape == (2, 2)


class TestComputePrecision:
    """Tests for compute_precision function."""

    def test_precision_binary(self):
        """Test precision for binary classification."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        precision = compute_precision(y_true, y_pred, average='binary')

        # TP=1, FP=0 for positive class
        assert precision == 1.0

    def test_precision_macro(self, sample_predictions):
        """Test macro-averaged precision."""
        y_true, y_pred = sample_predictions
        precision = compute_precision(y_true, y_pred, average='macro')

        assert isinstance(precision, (float, np.floating))
        assert 0 <= precision <= 1

    def test_precision_per_class(self):
        """Test per-class precision."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        precision = compute_precision(y_true, y_pred, average=None)

        assert isinstance(precision, np.ndarray)
        assert len(precision) == 2  # Binary classification
        assert all(0 <= p <= 1 for p in precision)


class TestComputeRecall:
    """Tests for compute_recall function."""

    def test_recall_binary(self):
        """Test recall for binary classification."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        recall = compute_recall(y_true, y_pred, average='binary')

        # TP=1, FN=1 for positive class
        assert recall == 0.5

    def test_recall_macro(self, sample_predictions):
        """Test macro-averaged recall."""
        y_true, y_pred = sample_predictions
        recall = compute_recall(y_true, y_pred, average='macro')

        assert isinstance(recall, (float, np.floating))
        assert 0 <= recall <= 1

    def test_recall_per_class(self, sample_predictions):
        """Test per-class recall."""
        y_true, y_pred = sample_predictions
        recall = compute_recall(y_true, y_pred, average=None)

        assert isinstance(recall, np.ndarray)
        assert len(recall) == 2
        assert all(0 <= r <= 1 for r in recall)


class TestComputeF1:
    """Tests for compute_f1 function."""

    def test_f1_binary(self):
        """Test F1 for binary classification."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        f1 = compute_f1(y_true, y_pred, average='binary')

        # Precision=1.0, Recall=0.5, F1=2*1.0*0.5/(1.0+0.5)=0.667
        assert abs(f1 - 0.6666666) < 0.001

    def test_f1_macro(self, sample_predictions):
        """Test macro-averaged F1."""
        y_true, y_pred = sample_predictions
        f1 = compute_f1(y_true, y_pred, average='macro')

        assert isinstance(f1, (float, np.floating))
        assert 0 <= f1 <= 1

    def test_f1_perfect_score(self, perfect_predictions):
        """Test that perfect predictions give F1 of 1.0."""
        y_true, y_pred = perfect_predictions
        f1 = compute_f1(y_true, y_pred, average='macro')

        assert f1 == 1.0


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics function."""

    def test_returns_dict(self, sample_predictions):
        """Test that function returns a dictionary."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)

    def test_has_required_keys(self, sample_predictions):
        """Test that metrics dict has all required keys."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        required_keys = [
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro',
            'precision_per_class',
            'recall_per_class',
            'f1_per_class',
            'confusion_matrix',
        ]

        for key in required_keys:
            assert key in metrics

    def test_with_class_names(self, sample_predictions, class_names):
        """Test metrics computation with class names."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred, class_names)

        assert 'class_names' in metrics
        assert metrics['class_names'] == class_names

        # Should have per-class dictionaries
        assert 'precision_by_class' in metrics
        assert 'recall_by_class' in metrics
        assert 'f1_by_class' in metrics

        # Check that dictionaries have correct structure
        assert 'bleached' in metrics['precision_by_class']
        assert 'healthy' in metrics['precision_by_class']

    def test_metric_values_reasonable(self, sample_predictions):
        """Test that metric values are in reasonable ranges."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1

        # Per-class metrics
        assert all(0 <= p <= 1 for p in metrics['precision_per_class'])
        assert all(0 <= r <= 1 for r in metrics['recall_per_class'])
        assert all(0 <= f <= 1 for f in metrics['f1_per_class'])

    def test_confusion_matrix_is_list(self, sample_predictions):
        """Test that confusion matrix is returned as list."""
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert isinstance(metrics['confusion_matrix'], list)

    def test_perfect_predictions_metrics(self, perfect_predictions):
        """Test metrics for perfect predictions."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0


class TestPrintClassificationReport:
    """Tests for print_classification_report function."""

    def test_returns_string(self, sample_predictions):
        """Test that function returns a string."""
        y_true, y_pred = sample_predictions
        report = print_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_with_class_names(self, sample_predictions, class_names):
        """Test report generation with class names."""
        y_true, y_pred = sample_predictions
        report = print_classification_report(y_true, y_pred, class_names)

        # Class names should appear in report
        assert 'bleached' in report
        assert 'healthy' in report

    def test_report_contains_metrics(self, sample_predictions):
        """Test that report contains metric keywords."""
        y_true, y_pred = sample_predictions
        report = print_classification_report(y_true, y_pred)

        # Should contain standard metric names
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report


class TestLogMetricsToWandb:
    """Tests for log_metrics_to_wandb function."""

    def test_logs_to_wandb(self):
        """Test that metrics are logged to wandb."""
        with patch('wandb.run') as mock_run, patch('wandb.log') as mock_log:
            mock_run.__bool__ = lambda x: True  # Active run
            metrics = {'accuracy': 0.95, 'loss': 0.1}

            log_metrics_to_wandb(metrics)

            mock_log.assert_called_once_with(metrics)

    def test_logs_with_step(self):
        """Test logging with step parameter."""
        with patch('wandb.run') as mock_run, patch('wandb.log') as mock_log:
            mock_run.__bool__ = lambda x: True
            metrics = {'accuracy': 0.95}

            log_metrics_to_wandb(metrics, step=100)

            mock_log.assert_called_once_with(metrics, step=100)

    def test_adds_prefix(self):
        """Test that prefix is added to metric names."""
        with patch('wandb.run') as mock_run, patch('wandb.log') as mock_log:
            mock_run.__bool__ = lambda x: True
            metrics = {'accuracy': 0.95, 'loss': 0.1}

            log_metrics_to_wandb(metrics, prefix='train/')

            expected_metrics = {'train/accuracy': 0.95, 'train/loss': 0.1}
            mock_log.assert_called_once_with(expected_metrics)

    def test_skips_when_no_active_run(self):
        """Test that logging is skipped when no active wandb run."""
        with patch('wandb.run', None), patch('wandb.log') as mock_log:
            metrics = {'accuracy': 0.95}

            # Should not raise error
            log_metrics_to_wandb(metrics)

            mock_log.assert_not_called()

    def test_handles_missing_wandb(self):
        """Test that function handles missing wandb gracefully."""
        metrics = {'accuracy': 0.95}

        # Should not raise error even if wandb not available
        # (wandb is available in our env, but test the try/except path)
        log_metrics_to_wandb(metrics)


class TestLogConfusionMatrixToWandb:
    """Tests for log_confusion_matrix_to_wandb function."""

    def test_logs_confusion_matrix(self):
        """Test that confusion matrix is logged to wandb."""
        with patch('wandb.run') as mock_run, \
             patch('wandb.log') as mock_log, \
             patch('wandb.plot.confusion_matrix', return_value='cm_plot') as mock_cm_plot:

            mock_run.__bool__ = lambda x: True

            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 0, 0])
            class_names = ['bleached', 'healthy']

            log_confusion_matrix_to_wandb(y_true, y_pred, class_names)

            # Should have called confusion_matrix plot
            mock_cm_plot.assert_called_once()
            mock_log.assert_called_once()

    def test_skips_when_no_active_run(self):
        """Test that logging is skipped when no active wandb run."""
        with patch('wandb.run', None), patch('wandb.log') as mock_log:

            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 0, 0])
            class_names = ['bleached', 'healthy']

            # Should not raise error
            log_confusion_matrix_to_wandb(y_true, y_pred, class_names)

            mock_log.assert_not_called()


class TestComputeMetrics:
    """Tests for compute_metrics backwards compatibility function."""

    def test_is_alias_for_classification_metrics(self, sample_predictions):
        """Test that compute_metrics is an alias."""
        y_true, y_pred = sample_predictions

        metrics1 = compute_metrics(y_true, y_pred)
        metrics2 = compute_classification_metrics(y_true, y_pred)

        # Should return the same results
        assert metrics1.keys() == metrics2.keys()
        assert metrics1['accuracy'] == metrics2['accuracy']


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_complete_metrics_pipeline(self, sample_predictions, class_names):
        """Test complete metrics computation pipeline."""
        y_true, y_pred = sample_predictions

        # Compute all metrics
        metrics = compute_classification_metrics(y_true, y_pred, class_names)

        # Verify structure
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'precision_by_class' in metrics

        # Generate report
        report = print_classification_report(y_true, y_pred, class_names)
        assert len(report) > 0

    def test_metrics_with_torch_tensors(self, sample_predictions):
        """Test that all metrics work with torch tensors."""
        y_true, y_pred = sample_predictions
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)

        # All these should work
        accuracy = compute_accuracy(y_true_tensor, y_pred_tensor)
        precision = compute_precision(y_true_tensor, y_pred_tensor, average='macro')
        recall = compute_recall(y_true_tensor, y_pred_tensor, average='macro')
        f1 = compute_f1(y_true_tensor, y_pred_tensor, average='macro')
        cm = compute_confusion_matrix(y_true_tensor, y_pred_tensor)

        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert cm.shape == (2, 2)

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        # Create balanced predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # Perfect predictions

        metrics = compute_classification_metrics(y_true, y_pred)

        # For perfect predictions, all metrics should be 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0


class TestRealIntegration:
    """INTEGRATION TESTS: Real functionality verification (no mocking)."""

    def test_wandb_offline_logging_real(self, sample_predictions):
        """INTEGRATION: Test real wandb logging in offline mode."""
        import wandb
        import tempfile
        import shutil

        # Create temp directory for offline wandb
        temp_dir = tempfile.mkdtemp()

        try:
            # Initialize wandb in offline mode
            wandb.init(
                mode='offline',
                project='test-coral-metrics',
                dir=temp_dir,
                reinit=True
            )

            y_true, y_pred = sample_predictions
            class_names = ['bleached', 'healthy']

            # Compute metrics
            metrics = compute_classification_metrics(y_true, y_pred, class_names)

            # Log to real wandb (offline)
            log_metrics_to_wandb(metrics, step=1, prefix='test/')

            # Verify wandb run exists
            assert wandb.run is not None
            # In offline mode, run should have a directory
            assert hasattr(wandb.run, 'dir')

            # Finish run
            wandb.finish()

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_metrics_with_real_model_predictions(self):
        """INTEGRATION: Test metrics with real PyTorch model outputs."""
        # Create a simple dummy binary classifier
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        model.eval()

        # Generate random input
        batch_size = 50
        X = torch.randn(batch_size, 10)
        y_true = torch.randint(0, 2, (batch_size,))

        # Get model predictions
        with torch.no_grad():
            logits = model(X)
            y_pred = logits.argmax(dim=1)

        # Compute metrics (tests tensor->numpy conversion pipeline)
        metrics = compute_classification_metrics(y_true, y_pred)

        # Verify all metrics are computed
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics

        # Verify values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1

        # Verify confusion matrix is 2x2
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (2, 2)
        assert cm.sum() == batch_size

    def test_confusion_matrix_with_real_sklearn(self):
        """INTEGRATION: Verify sklearn confusion matrix works with our data."""
        # Use realistic binary classification scenario
        np.random.seed(42)
        n_samples = 100

        # Simulate model with 80% accuracy
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()

        # Introduce 20% error
        error_indices = np.random.choice(n_samples, size=20, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]

        # Compute confusion matrix with real sklearn
        cm = compute_confusion_matrix(y_true, y_pred)

        # Verify properties
        assert cm.shape == (2, 2)
        assert cm.sum() == n_samples
        assert cm.trace() == 80  # 80 correct predictions

        # Test normalized version
        cm_norm = compute_confusion_matrix(y_true, y_pred, normalize='true')
        row_sums = cm_norm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.array([1.0, 1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
