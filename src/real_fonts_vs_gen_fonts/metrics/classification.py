from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryCalibrationError

ECE_N_BINS: int = 15


def make_scorer_metrics(prefix: str = "") -> MetricCollection:
    """Scorer 評価用の MetricCollection を返す。

    Args:
        prefix: メトリクス名のプレフィックス (例: "val/" → "val/accuracy")。
    Returns:
        MetricCollection(accuracy, auroc, ece)。

    Usage:
        self.val_metrics = make_scorer_metrics("val/")
        self.val_metrics.update(scores, labels)   # validation_step 内
        self.log_dict(self.val_metrics.compute())  # epoch_end 内
        self.val_metrics.reset()
    """
    return MetricCollection(
        {
            f"{prefix}accuracy": BinaryAccuracy(),
            f"{prefix}auroc": BinaryAUROC(),
            f"{prefix}ece": BinaryCalibrationError(n_bins=ECE_N_BINS, norm="l1"),
        }
    )
