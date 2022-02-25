# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from ..models.experimental import attempt_load
from ..data.datasets import create_dataloader
from ..utils.general import (
    coco80_to_coco91_class,
    check_dataset,
    check_img_size,
    check_suffix,
    box_iou,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    xywh2xyxy,
    increment_path,
    colorstr,
)
from ..utils.metrics import ap_per_class, ConfusionMatrix
from ..utils.plots import output_to_target, plot_images
from ..utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (
            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        )  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


@torch.no_grad()
class Yolov5Evaluator:
    def __init__(
        self,
        data,
        conf_thres=0.001,
        iou_thres=0.6,
        device="",
        single_cls=False,
        augment=False,
        verbose=False,
        project="runs/val",
        name="exp",
        exist_ok=False,
        half=True,
        save_dir=Path(""),
        plots=True,
    ) -> None:
        self.data = check_dataset(data)  # check
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IoU threshold
        self.device = device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.single_cls = single_cls  # treat as single-class dataset
        self.augment = augment  # augmented inference
        self.verbose = verbose  # verbose output
        self.project = project  # save to project/name
        self.name = name  # save to project/name
        self.exist_ok = exist_ok  # existing project/name ok, do not increment
        self.half = half  # use FP16 half-precision inference
        self.save_dir = save_dir
        self.plots = plots

        self.nc = 1 if self.single_cls else int(self.data["nc"])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.dt = [0.0, 0.0, 0.0]
        self.names = {k: v for k, v in enumerate(self.data["names"])}
        self.s = ("%20s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )

        # coco stuff
        self.is_coco = isinstance(self.data.get("val"), str) and self.data[
            "val"
        ].endswith(
            "coco/val2017.txt"
        )  # COCO dataset
        self.class_map = coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.jdict = []
        self.iou_thres = 0.65 if self.is_coco else self.iou_thres

        # metric stuff
        self.seen = 0
        self.stats = []
        self.total_loss = torch.zeros(3)
        self.metric = Metric()

    def run_training(self, model, dataloader, compute_loss=None):
        """This is for evaluation when training."""
        self.device = next(model.parameters()).device  # get model device
        # self.iouv.to(self.device)
        self.total_loss = torch.zeros(3, device=self.device)
        self.half &= self.device.type != "cpu"  # half precision only supported on CUDA
        model.half() if self.half else model.float()
        # Configure
        model.eval()

        # inference
        for batch_i, (img, targets, paths, shapes, _) in enumerate(
            tqdm(dataloader, desc=self.s)
        ):
            targets = targets.to(self.device)
            out = self.inference(model, img, targets, compute_loss)

            # Statistics per image
            for si, pred in enumerate(out):
                self.seen += 1
                predn = pred.clone()

                # I tested `compute_stat` and `compute_stat_native`,
                # it shows the same results.
                # but maybe I didn't do enough experiments, 
                # so I left the related code(`compute_stat_native`).
                self.compute_stat(si, predn, targets)
                # shape = shapes[si][0]
                # ratio_pad = shapes[si][1]
                # self.compute_stat_native(si, img, predn, targets, shape, ratio_pad)

            self.plot_images(batch_i, img, targets, out, paths)

        # compute map and print it.
        t = self.after_infer()

        # Return results
        model.float()  # for training
        return (
            (
                *self.metric.results(),
                *(self.total_loss.cpu() / len(dataloader)).tolist(),
            ),
            self.metric.get_maps(self.nc),
            t,
        )

    def run(
        self,
        weights,
        batch_size,
        imgsz,
        save_txt=False,
        save_conf=False,
        save_json=False,
        task="val",
    ):
        """This is for native evaluation."""
        model, dataloader, imgsz = self.before_infer(weights, batch_size, imgsz, save_txt, task)
        # self.iouv.to(self.device)
        self.half &= self.device.type != "cpu"  # half precision only supported on CUDA
        model.half() if self.half else model.float()
        # Configure
        model.eval()

        # inference
        for batch_i, (img, targets, paths, shapes, _) in enumerate(
            tqdm(dataloader, desc=self.s)
        ):
            targets = targets.to(self.device)
            out = self.inference(model, img, targets)

            # Statistics per image
            for si, pred in enumerate(out):
                self.seen += 1
                path = Path(paths[si])
                predn = pred.clone()
                shape = shapes[si][0]

                # I tested `compute_stat` and `compute_stat_native`,
                # it shows the same results.
                # but maybe I didn't do enough experiments, 
                # so I left the related code(`compute_stat_native`).
                self.compute_stat(si, predn, targets)
                # ratio_pad = shapes[si][1]
                # self.compute_stat_native(si, img, predn, targets, shape, ratio_pad)

                # Save/log
                if save_txt:
                    save_one_txt(
                        pred,
                        save_conf,
                        shape,
                        file=self.save_dir / "labels" / (path.stem + ".txt"),
                    )
                if save_json:
                    save_one_json(
                        pred, self.jdict, path, self.class_map
                    )  # append to COCO-JSON dictionary

            self.plot_images(batch_i, img, targets, out, paths)

        # compute map and print it.
        t = self.after_infer()

        # Print speeds
        shape = (batch_size, 3, imgsz, imgsz)
        print(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}"
            % t
        )

        s = (
            f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        # Return results
        return (
            (
                *self.metric.results(),
                *(self.total_loss.cpu() / len(dataloader)).tolist(),
            ),
            self.metric.get_maps(self.nc),
            t,
        )

    def before_infer(self, weights, batch_size, imgsz, save_txt, task="val"):
        "prepare for evaluation without training."
        self.device = select_device(self.device, batch_size=batch_size)

        # Directories
        self.save_dir = increment_path(
            Path(self.project) / self.name, exist_ok=self.exist_ok
        )  # increment run
        (self.save_dir / "labels" if save_txt else self.save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        check_suffix(weights, ".pt")
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        if self.device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(self.device)
                .type_as(next(model.parameters()))
            )  # run once
        pad = 0.0 if task == "speed" else 0.5
        task = (
            task if task in ("train", "val", "test") else "val"
        )  # path to train/val/test images
        dataloader = create_dataloader(
            self.data[task],
            imgsz,
            batch_size,
            gs,
            self.single_cls,
            pad=pad,
            rect=True,
            prefix=colorstr(f"{task}: "),
        )[0]
        return model, dataloader, imgsz

    def inference(self, model, img, targets, compute_loss=None):
        """Inference"""
        t1 = time_sync()
        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Run model
        out, train_out = model(
            img, augment=self.augment
        )  # inference and training outputs
        self.dt[1] += time_sync() - t2

        # Compute loss
        if compute_loss:
            self.total_loss += compute_loss([x.float() for x in train_out], targets)[
                1
            ]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            self.device
        )  # to pixels
        t3 = time_sync()
        out = non_max_suppression(
            out,
            self.conf_thres,
            self.iou_thres,
            multi_label=True,
            agnostic=self.single_cls,
        )
        self.dt[2] += time_sync() - t3
        return out

    def after_infer(self):
        """Do something after inference, such as plots and get metrics.
        Return:
            t(tuple): speeds of per image.
        """
        # Plot confusion matrix
        if self.plots:
            self.confusion_matrix.plot(
                save_dir=self.save_dir, names=list(self.names.values())
            )

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(
                *stats, plot=self.plots, save_dir=self.save_dir, names=self.names
            )
            self.metric.set(p, r, ap, f1, ap_class)
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=self.nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # make this empty, cause make `stats` self is for reduce some duplicated codes.
        self.stats = []
        # print information
        self.print_metric(nt, stats)
        t = tuple(x / self.seen * 1e3 for x in self.dt)  # speeds per image
        return t

    def process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(
            detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
        )
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
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
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def compute_stat(self, si, predn, targets):
        """Compute states about ious. with boxs size in training img-size space."""
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(predn) == 0:
            if nl:
                self.stats.append(
                    (
                        torch.zeros(0, self.niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            return

        # Predictions
        if self.single_cls:
            predn[:, 5] = 0

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = self.process_batch(predn, labelsn, self.iouv)
            if self.plots:
                self.confusion_matrix.process_batch(predn, labelsn)
        else:
            correct = torch.zeros(predn.shape[0], self.niou, dtype=torch.bool)
        self.stats.append(
            (
                correct.cpu(),
                predn[:, 4].cpu(),
                predn[:, 5].cpu(),
                tcls,
            )
        )  # (correct, conf, pcls, tcls)

    def compute_stat_native(self, si, img, predn, targets, shape, ratio_pad):
        """Compute states about ious. with boxs size in native space."""
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(predn) == 0:
            if nl:
                self.stats.append(
                    (
                        torch.zeros(0, self.niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            return

        # Predictions
        if self.single_cls:
            predn[:, 5] = 0
        scale_coords(
            img[si].shape[1:], predn[:, :4], shape, ratio_pad
        )  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(
                img[si].shape[1:], tbox, shape, ratio_pad
            )  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = self.process_batch(predn, labelsn, self.iouv)
            if self.plots:
                self.confusion_matrix.process_batch(predn, labelsn)
        else:
            correct = torch.zeros(predn.shape[0], self.niou, dtype=torch.bool)
        self.stats.append(
            (
                correct.cpu(),
                predn[:, 4].cpu(),
                predn[:, 5].cpu(),
                tcls,
            )
        )  # (correct, conf, pcls, tcls)

    def print_metric(self, nt, stats):
        # Print results
        pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
        print(
            pf
            % (
                "all",
                self.seen,
                nt.sum(),
                self.metric.mp,
                self.metric.mr,
                self.metric.map50,
                self.metric.map,
            )
        )

        # Print results per class
        if self.verbose and self.nc > 1 and len(stats):
            for i, c in enumerate(self.metric.ap_class_index):
                print(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        nt[c],
                        self.metric.p[i],
                        self.metric.r[i],
                        self.metric.ap50[i],
                        self.metric.ap[i],
                    )
                )

    def plot_images(self, i, img, targets, out, paths):
        if (not self.plots) or i >= 3:
            return
        f = self.save_dir / f"val_batch{i}_labels.jpg"  # labels
        Thread(
            target=plot_images,
            args=(img, targets, paths, f, self.names),
            daemon=True,
        ).start()
        f = self.save_dir / f"val_batch{i}_pred.jpg"  # predictions
        Thread(
            target=plot_images,
            args=(img, output_to_target(out), paths, f, self.names),
            daemon=True,
        ).start()


class Metric:
    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

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

    def results(self):
        """return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def set(self, p, r, all_ap, f1, ap_class_index):
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index
