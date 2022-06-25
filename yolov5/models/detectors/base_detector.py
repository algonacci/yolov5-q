from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..backbones import build_backbone
from ..heads import build_head
from ..necks import build_neck
from torch import nn


@DETECTORS.register()
class BaseDetector(nn.Module):
    def __init__(
        self,
        backbone,
        neck,
        head,
    ) -> None:
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_test(self, img):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        output = self.neck(feat)
        return output
