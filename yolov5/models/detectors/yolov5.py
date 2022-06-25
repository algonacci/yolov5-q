from ..builder import DETECTORS
from .base_detector import BaseDetector


@DETECTORS.register()
class YOLOV5(BaseDetector):
    def __init__(self, backbone, neck, head) -> None:
        super(YOLOV5, self).__init__(backbone, neck, head)
