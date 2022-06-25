from ..builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn


@DETECTORS.register()
class BaseDetector(nn.Module):
    def __init__(self, backbone, neck, head,) -> None:
        super().__init__()
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
        output = self.head(feat)
        return output

    def forward_train(self, img):
        pass

    def forward(self, img):
        return self.forward_train(img) if self.training else self.forward_test(img)
