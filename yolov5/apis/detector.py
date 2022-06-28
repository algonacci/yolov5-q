from ..utils.torch_utils import select_device
from ..models.experimental import attempt_load
from ..data.augmentations import letterbox
from ..utils.boxes import non_max_suppression, scale_coords
from ..utils.timer import Timer
from ..utils.segment import (
    non_max_suppression_masks,
    process_mask_upsample,
    scale_masks,
)
from ..utils.plots import Visualizer, plot_masks, colors
from ..utils.torch_utils import is_parallel
from ..utils.checker import check_img_size
from ..utils.general import to_2tuple
import os
import numpy as np
import torch
import cv2


class Yolov5:
    """Yolov5 detection, support multi image and one model with fp16 inference."""

    def __init__(self, weights, device, img_hw=(384, 640), auto=False) -> None:
        device = select_device(device)
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        img_hw = to_2tuple(img_hw) if isinstance(img_hw, int) else img_hw

        self.auto = auto
        # inference hw
        self.img_hw = check_img_size(img_hw, s=stride)
        # original hw
        self.ori_hw = []

        self.device = device
        self.model = model.half() if self.device != "cpu" else model
        self.names = self.model.names
        self.vis = Visualizer(names=self.names)

        # time stuff
        self.timer = Timer(start=False, cuda_sync=True, round=1, unit="ms")
        self.times = {}

    def warmup(self):
        # Warmup model by running inference once
        # only warmup GPU models
        if self.device != "cpu":
            im = torch.zeros(1, 3, *self.img_hw).to(self.device).half()  # input image
            self.model(im)  # warmup

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
        self.current_hw = resized_img.shape[:2]

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

    def postprocess_one_model(
        self, preds, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False
    ):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            preds (torch.Tensor): [B, num_boxes, classes+5].
            conf_thres (float): confidence threshold.
            iou_thres (float): iou threshold.
            classes (None | List[int]): class filter according class index.
            agnostic (bool): Whether to do nms as there is only one class.
        Return:
            outputs (List[torch.Tensor]): List[torch.Tensor(num_boxes, 6)]xB.
        """
        outputs = non_max_suppression(
            preds,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic,
        )
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(self.current_hw, det[:, :4], self.ori_hw[i]).round()
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
        self.timer.start(reset=True)
        imgs = preprocess(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half()
        imgs /= 255.0
        self.times["preprocess"] = self.timer.since_last_check()
        preds = self.model(imgs)[0]
        self.times["inference"] = self.timer.since_last_check()
        outputs = self.postprocess_one_model(preds, conf_thres, iou_thres, classes)
        self.times["postprocess"] = self.timer.since_last_check()
        self.times["total"] = self.timer.since_start()
        self.ori_hw.clear()
        return outputs

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
        if images is a ndarray, then return ndarray.
        """
        return self.vis(images, outputs, vis_confs)


class Yolov5Segment(Yolov5):
    def __init__(self, weights, device, img_hw=(384, 640), auto=False) -> None:
        super(Yolov5Segment, self).__init__(weights, device, img_hw, auto)
        det = (
            self.model.module.model[-1]
            if is_parallel(self.model)
            else self.model.model[-1]
        )  # Detect() module
        self.mask_dim = det.mask_dim

    @torch.no_grad()
    def inference(
        self,
        images,
        conf_thres=0.4,
        iou_thres=0.5,
        classes=None,
        agnostic=False,
        areas=None,
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
        self.timer.start(reset=True)
        imgs = preprocess(images)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half()
        self.imgs = imgs / 255.0  # `self.imgs` is for faster ploting masks
        self.times["preprocess"] = self.timer.since_last_check()
        preds, out = self.model(self.imgs)
        proto = out[1]
        self.times["inference"] = self.timer.since_last_check()
        outputs, masks = self.postprocess_one_model(
            preds, proto, conf_thres, iou_thres, classes, agnostic
        )
        self.times["postprocess"] = self.timer.since_last_check()
        self.times["total"] = self.timer.since_start()
        self.ori_hw.clear()
        return outputs, masks

    def postprocess_one_model(
        self, preds, proto, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False
    ):
        """Postprocess multi images. NMS and scale coords to original image size.

        Args:
            preds (torch.Tensor): bbox+conf+mask+cls, [B, num_boxes, classes+5].
            proto (torch.Tensor): mask proto, [B, mask_dim, mask_h, mask_w].
            conf_thres (float): confidence threshold.
            iou_thres (float): iou threshold.
            classes (None | List[int]): class filter according class index.
            agnostic (bool): Whether to do nms as there is only one class.
        Return:
            outputs (List[torch.Tensor]): bbox+conf+cls, List[torch.Tensor(num_boxes, 6)]xB,
                in original hw space.
            masks (List[torch.Tensor]): binary mask, List[torch.Tensor(num_boxes, img_h, img_w)]xB.
                in input hw space.
        """
        outputs = non_max_suppression_masks(
            preds,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic,
            mask_dim=self.mask_dim,
        )
        out_masks = []
        for i, det in enumerate(outputs):  # detections per image
            if det is None or len(det) == 0:
                continue
            # mask stuff
            masks_conf = det[:, 6:]
            # binary mask, (img_h, img_w, n)
            masks = process_mask_upsample(proto[i], masks_conf, det[:, :4], self.current_hw)
            # n, img_h, img_w
            masks = masks.permute(2, 0, 1).contiguous()
            out_masks.append(masks)
            # bbox stuff
            det = det[:, :6]  # update the value in outputs, remove mask part.
            det[:, :4] = scale_coords(self.current_hw, det[:, :4], self.ori_hw[i]).round()
        return outputs, out_masks

    def visualize(self, images, outputs, out_masks, vis_confs=0.4):
        """Image visualize
        if images is a List of ndarray, then will return a List.
        if images is a ndarray, then return ndarray.
        Args:
            outputs: bbox+conf+cls, List[torch.Tensor(num_boxes, 6)]xB.
            masks: binary masks, List[torch.Tensor(num_boxes, img_h, img_w)]xB.
        """
        ori_type = type(images)
        # get original shape, cause self.ori_hw will be cleared
        images = images if isinstance(images, list) else [images]
        ori_hw = [img.shape[:2] for img in images]
        # init the list to keep image with masks.
        # TODO: fix this bug when output is empty.
        masks_images = []
        # draw masks
        for i, output in enumerate(outputs):
            if output is None or len(output) == 0:
                continue
            idx = output[:, 4] > vis_confs
            masks = out_masks[i][idx]
            mcolors = [colors(int(cls)) for cls in output[:, 5]]
            # NOTE: this way to draw masks is faster,
            # from https://github.com/dbolya/yolact
            # image with masks, (img_h, img_w, 3)
            img_masks = plot_masks(self.imgs[i], masks, mcolors)
            # scale image to original hw
            img_masks = scale_masks(self.imgs[i].shape[1:], img_masks, ori_hw[i])
            masks_images.append(img_masks)
        # TODO: make this(ori_type stuff) clean
        images = masks_images[0] if (len(masks_images) == 1) and type(masks_images) != ori_type else images[0]
        return self.vis(images, outputs, vis_confs)
