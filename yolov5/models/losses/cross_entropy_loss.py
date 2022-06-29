import torch
import torch.nn.functional as F
from torch import nn
from ..builder import LOSSES


def cross_entropy(
    pred,
    label,
    reduction="mean",
    class_weight=None,
    ignore_index=-100,
):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction=reduction, ignore_index=ignore_index
    )
    return loss


def binary_cross_entropy(
    pred,
    label,
    reduction="mean",
    class_weight=None,
    ignore_index=-100,
):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """

    if pred.dim() != label.dim():
        target = torch.full_like(pred, 0, device=pred.device)  # targets
        n = len(target)
        target[range(n), label] = 1
    else:
        target = label
    # TODO: add ignore_index
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.binary_cross_entropy_with_logits(
        pred, label, pos_weight=class_weight, reduction=reduction
    )
    return loss


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
    ) -> None:
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(
        self,
        cls_score,
        label,
    ):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
        Returns:
            torch.Tensor: The calculated loss.
        """
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device
            )
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, class_weight=class_weight, reduction=self.reduction
        )
        return loss_cls
