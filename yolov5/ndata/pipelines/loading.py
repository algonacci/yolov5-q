from ..builder import PIPELINES
import cv2
import numpy as np


@PIPELINES.register_module()
class LoadImageFromFile:
    def __init__(self, to_float32=False) -> None:
        self.to_float32 = to_float32

    def __call__(self, results):
        img_file = results["img_file"]
        img = cv2.imread(img_file)
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        return results


@PIPELINES.register_module()
class LoadAnnotations:
    def __init__(
        self,
        with_bbox=True,
        with_seg=False,
        denorm_bbox=False,
    ) -> None:
        self.with_bbox = with_bbox
        self.with_seg = with_seg
        self.denorm_bbox = denorm_bbox

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def _load_bboxes(self, results):
        pass
