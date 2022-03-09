from ..utils.torch_utils import select_device
from ..models.experimental import attempt_load
from ..data.augmentations import letterbox
from ..utils.boxes import non_max_suppression, scale_coords
from ..utils.plots import Visualizer
import os
import numpy as np
import torch


class Yolov5:
    """Yolov5 detection, support multi image and one model with fp16 inference."""

    def __init__(self, weights, device, img_hw, auto=False) -> None:
        self.model = attempt_load(weights)
        self.model = self.model.half()
        self.device = select_device(device)
        self.auto = auto
        self.img_hw = img_hw
        self.vis = Visualizer(names=self.model.names)

    def preprocess_one_img(self, image):
        """Preprocess one image.

        Args:
            image (numpy.ndarray | str): Input image or image path.
            auto (bool, optional): Whether to use rect.
        Return:
            resized_img (numpy.ndarray): Image after resize and transpose,
                (H, W, C) -> (1, C, H, W).
        """
        if type(image) == str and os.path.isfile(image):
            img_raw = cv2.imread(image)
        else:
            img_raw = image
        resized_img, _, _ = letterbox(
            img_raw,
            new_shape=self.img_hw,
            auto=self.auto,
            scaleFill=False,
        )
        if self.auto:
            self.img_hw = resized_img.shape[:2]

        # H, W, C -> 1, C, H, W
        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)[None, :]
        resized_img = np.ascontiguousarray(resized_img)
        self.ori_hw.append(img_raw.shape[:2])
        return resized_img

    def preprocess_multi_img(self, images):
        """Preprocess multi image.

        Args:
            images (List[numpy.ndarray] | List[str]): Input images or image paths.
            auto (bool, optional): Whether to use rect.
        Return:
            imgs (numpy.ndarray): Images after resize and transpose,
                List[(H, W, C)] -> (B, C, H, W).
        """

        resized_imgs = []
        for image in images:
            img = self.preprocess_one_img(image)
            resized_imgs.append(img)
        imgs = np.concatenate(resized_imgs, axis=0)
        return imgs

    def postprocess_one_model(self, preds, conf_thres=0.4, iou_thres=0.5, classes=None):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            preds (torch.Tensor): [B, num_boxes, classes+5].
            conf_thres (float): confidence threshold.
            iou_thres (float): iou threshold.
            classes (None | List[int]): class filter according class index.
        Return:
            outputs (List[torch.Tensor]): List[torch.Tensor(num_boxes, 6)]xB.
        """
        outputs = non_max_suppression(
            preds, conf_thres, iou_thres, classes=classes, agnostic=False
        )
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(
                self.img_hw, det[:, :4], self.ori_hw[i], scale_fill=False
            ).round()
        return outputs

    @torch.no_grad()
    def inference(
        self, images, conf_thres=0.4, iou_thres=0.5, classes=None, areas=None
    ):
        """Inference.

        Args:
            image (np.ndarray | List[np.ndarray] | str | List[str]): Input image, input images or 
                image path or image paths.
            conf_thres (float): confidence threshold.
            iou_thres (float): iou threshold.
            classes (None | List[int]): class filter according class index.
        Return:
            outputs (List[torch.Tensor]): List[torch.Tensor(num_boxes, 6)]xB.
        """
        preprocess = (
            self.preprocess_multi_img
            if isinstance(images, list)
            else self.preprocess_one_img
        )
        imgs = preprocess(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half()
        imgs /= 255.0
        preds = self.model(imgs)[0]
        output = self.postprocess_one_model(preds, conf_thres, iou_thres, classes)
        self.ori_hw.clear()
        return output

    def __call__(self, images):
        """This is a simplify inference.
        Args:
            image (np.ndarray | List[np.ndarray] | str | List[str]): Input image, input images or 
                image path or image paths.
        """
        return self.inference(images)

    def visualize(self, images, outputs, vis_confs=0.4):
        """Image visualize
        if images is a List of ndarray, then will return a List.
        if images is a ndarray , then return ndarray.
        """
        return self.vis(images, outputs, vis_confs)


class Yolov5Segment(Yolov5):
    def __init__(self, weights, device, img_hw, auto=False) -> None:
        super(Yolov5Segment, self).__init__(weights, device, img_hw, auto)


if __name__ == "__main__":
    from tqdm import tqdm
    import cv2

    detector = Yolov5(weights="./weights/yolov5n.pt", device=0, img_hw=(384, 640))

    cap = cv2.VideoCapture("/e/1.avi")
    frames = 10000
    pbar = tqdm(range(frames), total=frames)

    for frame_num in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = detector(frame)
        detector.visualize(frame, outputs)
        cv2.imshow("p", frame)
        if cv2.waitKey(1) == ord("q"):
            break
