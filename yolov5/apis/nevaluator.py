import torch
import numpy as np
import torch.nn.functional as F
from lqcv.bbox.convert import xywh2xyxy
from tqdm import tqdm
from ..utils.metrics import box_iou
from ..utils.metrics import ConfusionMatrix
from ..utils.segment import (
    non_max_suppression_masks,
    mask_iou,
    process_mask,
    process_mask_upsample,
    scale_masks,
)

def evaluation_on_dataset(model, dataloader, evaluator):
    for i, (imgs, targets) in tqdm(dataloader, desc=""):
        imgs = (
            imgs.cuda().float() / 255.0
        )  # uint8 to float32, 0-255 to 0.0-1.0
        preds = model(imgs)


class DetectionEvaluator:
    def __init__(
        self,
        conf_thres=0.001,
        iou_thres=0.6,
        single_cls=False,
    ) -> None:
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        # self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self._status = []
        self.single_cls = single_cls

    def reset(self):
        self._status = []

    def process(self, outputs, targets):
        labels = targets["labels"]

        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        if len(outputs) == 0:
            if nl:
                self.stats.append(
                    (
                        torch.zeros(0, self.niou, dtype=torch.bool),  # boxes
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            return

        # Predictions
        if self.single_cls:
            outputs[:, 5] = 0
        # Evaluate
        if nl:
            # boxes
            correct = self.process_batch(outputs, targets)
            # if self.plots:
            #     self.confusion_matrix.process_batch(predn, labelsn)
        else:
            correct = torch.zeros(outputs.shape[0], self.niou, dtype=torch.bool)
        self.stats.append(
            (
                correct.cpu(),
                outputs[:, 4].cpu(),
                outputs[:, 5].cpu(),
                tcls,
            )
        )  # (correct, conf, pcls, tcls)

    def _process_batch(self, outputs, targets):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            outputs (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        labels = targets["labels"]

        pred_bboxes = outputs[:, :4]
        pred_cls = outputs[:, 5]

        gt_bboxes = labels[:, 1:]
        gt_cls = labels[:, 0:1]

        gt_bboxes = xywh2xyxy(gt_bboxes)  # target boxes

        correct = torch.zeros(
            outputs.shape[0],
            self.iouv.shape[0],
            dtype=torch.bool,
            device=self.iouv.device,
        )
        iou = box_iou(gt_bboxes, pred_bboxes)
        x = torch.where(
            (iou >= self.iouv[0]) & (gt_cls == pred_cls)
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(self.iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= self.iouv
        return correct

    def evaluate(self):
        """Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory.
            prefix: prefix.
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        tp, conf, pred_cls, target_cls = [
            np.concatenate(x, 0) for x in zip(*self._stats)
        ]  # to numpy
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = (
            np.zeros((nc, tp.shape[1])),
            np.zeros((nc, 1000)),
            np.zeros((nc, 1000)),
        )
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(
                    -px, -conf[i], recall[:, 0], left=0
                )  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(
                    -px, -conf[i], precision[:, 0], left=1
                )  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self._compute_ap(
                        recall[:, j], precision[:, j]
                    )

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        names = [
            v for k, v in names.items() if k in unique_classes
        ]  # list: only classes that have data
        names = {i: v for i, v in enumerate(names)}  # to dict

        i = f1.mean(0).argmax()  # max F1 index
        return Metric(p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32"))

    def _compute_ap(self, recall, precision):
        """Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = "interp"  # methods: 'continuous', 'interp'
        if method == "interp":
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[
                0
            ]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec


class SegmentEvaluator(DetectionEvaluator):
    def __init__(
        self,
        conf_thres=0.001,
        iou_thres=0.6,
        single_cls=False,
        mask_ratio=1,
        training=False,
    ) -> None:
        super().__init__(conf_thres, iou_thres, single_cls)
        self.mask_downsample_ratio = mask_ratio
        self.training = training

    def _process_batch(self, outputs, targets):
        pred = outputs["detections"]
        mask_atten = outputs["coefficients"]
        prototype = outputs["prototype"]

        labels = targets["labels"]
        gt_masks = targets["masks"]

        correct = torch.zeros(
            pred.shape[0],
            self.iouv.shape[0],
            dtype=torch.bool,
            device=self.iouv.device,
        )
        pred_masks = self._get_predmasks(
            mask_atten, prototype, labels[:, 1:], gt_masks.shape[1:]
        )

        if not self.plots:
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(0),
                pred_masks.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        iou = mask_iou(
            gt_masks.view(gt_masks.shape[0], -1),
            pred_masks.view(pred_masks.shape[0], -1),
        )
        x = torch.where(
            (iou >= self.iouv[0]) & (labels[:, 0:1] == pred[:, 5])
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(self.iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= self.iouv
        return correct

    def _get_predmasks(self, mask_atten, prototype, bboxes, gt_shape):
        """Get pred masks in different ways.
        1. process_mask, for val when training, eval with low quality(1/mask_ratio of original size)
            mask for saving cuda memory.
        2. process_mask_upsample, for val after training to get high quality mask(original size).

        Args:
            mask_atten(torch.Tensor): mask coefficients, (N, mask_dim).
            prototype(torch.Tensor): output of mask prototype, (mask_dim, mask_h, mask_w).
            bboxes(torch.Tensor): output of network, (N, 4).
            gt_shape(tuple): shape of gt mask, this shape may not equal to input size of
                input image, Cause the mask_downsample_ratio.
        Return:
            pred_mask(torch.Tensor): predition of final masks with the same size with
                input image, (N, input_h, input_w).
        """
        process = process_mask if self.training else process_mask_upsample
        gt_shape = (
            gt_shape[0] * self.mask_downsample_ratio,
            gt_shape[1] * self.mask_downsample_ratio,
        )
        # n, h, w
        pred_mask = (
            process(prototype, mask_atten, bboxes, shape=gt_shape)
            .permute(2, 0, 1)
            .contiguous()
        )
        return pred_mask


class Metric:
    def __init__(self, p=[], r=[], f1=[], all_ap=[], ap_class_index=[]) -> None:
        self.p = p  # (nc, )
        self.r = r  # (nc, )
        self.f1 = f1  # (nc, )
        self.all_ap = all_ap  # (nc, 10)
        self.ap_class_index = ap_class_index  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.metric_box.ap_class_index
