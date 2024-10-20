from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

__all__ = [
    "BinaryCrossEntropyLoss",
    "UniformOffLabelsBCEWithLogitsLoss",
]


class BinaryCrossEntropyLoss(nn.Module):
    """BCE with optional one-hot from dense targets, label smoothing, thresholding
    from https://github.com/rwightman/pytorch-image-models/blob/a520da9b49/timm/loss/binary_cross_entropy.py

    The label smoothing is done as in `torch.nn.CrossEntropyLoss`.
    In other words, the formula from https://arxiv.org/abs/1512.00567 is strictly followed
    even if input targets samples are sparse, unlike in timm.

    Important: Inputs are assumed to be logits. Targets can be either dense or sparse, and in the latter
    they should not be in logit space.
    """

    def __init__(
        self,
        smoothing=0.0,
        target_threshold: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(BinaryCrossEntropyLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        num_classes = x.shape[-1]

        # convert dense to sparse
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)

        # apply smoothing if needed
        if self.smoothing > 0.0:
            # just like in `torch.nn.CrossEntropyLoss`
            target = target * (1 - self.smoothing) + self.smoothing / num_classes

        # Make target 0, or 1 if threshold set
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        return F.binary_cross_entropy_with_logits(
            x, target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        result = f"reduction={self.reduction}, "
        if self.smoothing > 0:
            result += f"smoothing={self.smoothing}, "
        if self.target_threshold is not None:
            result += f"target_threshold={self.target_threshold}, "
        if self.weight is not None:
            result += f"weight={self.weight.shape}, "
        if self.pos_weight is not None:
            result += f"pos_weight={self.pos_weight.shape}, "
        result = result[:-2]
        return result


class UniformOffLabelsBCEWithLogitsLoss(nn.Module):
    """
    BCE loss with off value targets equal to some value.
    If not provided then it is `1/N`, where `N` is the number of classes.
    The on values are set to 1 as normal.

    This is best explained with an example, as follows:

    Examples
    --------
    Let N=5 and our target be t=3. Then t will be mapped to the following:
    `[0.2, 0.2, 0.2, 1.0, 0.2]`.

    If a particular off value is provided instead for example 2e-3 then it's:
    `[2e-3, 2e-3, 2e-3, 1.0, 2e-3]`
    """

    def __init__(self, reduction: str = "mean", off_label: Optional[float] = None):
        super().__init__()
        self.reduction = reduction
        self.off_label = off_label

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        num_classes = x.shape[-1]
        off_value = self.off_label or (1.0 / num_classes)
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)

        # make off values (0) to at least 1/N
        target = target.clamp(min=off_value)

        return F.binary_cross_entropy_with_logits(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        result = f"reduction={self.reduction}, "
        if self.off_label is not None:
            result += f"off_label={self.off_label}, "
        result = result[:-2]
        return result

class MultiLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes, threshold, only=None):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))
        self.only = only
    
    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if labels[i][j] == 1.0:
                        if batch_sample_logits[j] >= self.threshold:
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0
                    else:
                        if batch_sample_logits[j] >= self.threshold:
                            self.false_positives += 1.0
                        else:
                            self.true_negatives += 1.0

    def compute(self):
        self.accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives +
                         self.true_negatives + self.false_positives + self.false_negatives))
        self.precision = (self.true_positives /
                          (self.true_positives + self.false_positives))
        self.recall = (self.true_positives /
                       (self.true_positives + self.false_negatives))
        self.f_score = ((2 * self.true_positives) / (2 * self.true_positives +
                        self.false_positives + self.false_negatives))
        ret_dict = {'Accuracy': self.accuracy, 'Precision': self.precision, 'Recall': self.recall, 'F-Score': self.f_score, 'True Positives': self.true_positives, 'True Negatives': self.true_negatives, 'False Positives': self.false_positives, 'False Negatives': self.false_negatives}

        return ret_dict if self.only is None else ret_dict[self.only]


    def save(self, model, classifier_type, dataset):
        f = open(model + "_" + classifier_type + "_" +
                 dataset + "_" + "test_metrics.txt", "w")
        f.write("Accuracy: " + str(self.accuracy.item()) + "\n")
        f.write("Precision: " + str(self.precision.item()) + "\n")
        f.write("Recall: " + str(self.recall.item()) + "\n")
        f.write("F-Score: " + str(self.f_score.item()))
        f.close()

class MLD(nn.Module):
    # TODO DOCUMENT THIS
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.reduction = "none"
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, student, teacher, *args):
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)

        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)

        return loss
