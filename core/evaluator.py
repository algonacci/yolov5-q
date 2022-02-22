# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class,
    check_dataset,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    box_iou,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    xywh2xyxy,
    set_logging,
    increment_path,
    colorstr,
    print_args,
)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


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
    def __init__(self, data) -> None:
        self.data = data
        self.weights = None  # model.pt path(s)
        self.batch_size = 32  # batch size
        self.imgsz = 640  # inference size (pixels)
        self.conf_thres = 0.001  # confidence threshold
        self.iou_thres = 0.6  # NMS IoU threshold
        self.task = "val"  # train, val, test, speed or study
        self.device = ""  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.single_cls = False  # treat as single-class dataset
        self.augment = False  # augmented inference
        self.verbose = False  # verbose output
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_json = False  # save a COCO-JSON results file
        self.project = ROOT / "runs/val"  # save to project/name
        self.name = "exp"  # save to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.half = True  # use FP16 half-precision inference
        self.model = None
        self.dataloader = None
        self.save_dir = Path("")
        self.plots = True
        self.compute_loss = None

        self.metric = Metric()

    def run(self):
        self.training = self.model is not None
        if self.training:  # called by train.py
            self.eval_training()
        else:  # called directly
            self.eval()

        # Half
        self.half &= self.device.type != "cpu"  # half precision only supported on CUDA
        self.model.half() if self.half else self.model.float()

        # Configure
        self.model.eval()
        is_coco = isinstance(self.data.get("val"), str) and self.data["val"].endswith(
            "coco/val2017.txt"
        )  # COCO dataset
        nc = 1 if self.single_cls else int(self.data["nc"])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10).to(
            self.device
        )  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        # initialization
        seen = 0
        self.confusion_matrix = ConfusionMatrix(nc=nc)
        names = {
            k: v
            for k, v in enumerate(
                self.model.names
                if hasattr(self.model, "names")
                else self.model.module.names
            )
        }
        class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
        s = ("%20s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        self.dt = [0.0, 0.0, 0.0]
        self.loss = torch.zeros(3, device=self.device)
        jdict, stats = [], []

        # inference
        for batch_i, (img, targets, paths, shapes) in enumerate(
            tqdm(self.dataloader, desc=s)
        ):
            tp, ti, tn, out = self.inference(img, targets)
            self.dt[0] += tp
            self.dt[1] += ti
            self.dt[1] += tn

            # Statistics per image
            for si, pred in enumerate(out):
                seen += 1
                predn = pred.clone()  # for native space
                path = Path(paths[si])
                shape = shapes[si][0]
                ratio_pad = shapes[si][1]

                stat = self.compute_stat(si, img, predn, targets, shape, ratio_pad)
                if stat is not None:
                    stats.append(stat)

                # Save/log
                if self.save_txt:
                    save_one_txt(
                        predn,
                        self.save_conf,
                        shape,
                        file=self.save_dir / "labels" / (path.stem + ".txt"),
                    )
                if self.save_json:
                    save_one_json(
                        predn, jdict, path, class_map
                    )  # append to COCO-JSON dictionary

            # Plot images
            if self.plots and batch_i < 3:
                f = self.save_dir / f"val_batch{batch_i}_labels.jpg"  # labels
                Thread(
                    target=plot_images,
                    args=(img, targets, paths, f, names),
                    daemon=True,
                ).start()
                f = self.save_dir / f"val_batch{batch_i}_pred.jpg"  # predictions
                Thread(
                    target=plot_images,
                    args=(img, output_to_target(out), paths, f, names),
                    daemon=True,
                ).start()

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(
                *stats, plot=self.plots, save_dir=self.save_dir, names=names
            )
            self.metric.set(p, r, ap, f1, ap_class)
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        t = tuple(x / seen * 1e3 for x in self.dt)  # speeds per image
        # print information
        self.print(seen, nt, nc, names, stats, t)

        # Plots
        if self.plots:
            self.confusion_matrix.plot(
                save_dir=self.save_dir, names=list(names.values())
            )

        # TODO
        # Save JSON
        if self.save_json and len(jdict):
            w = (
                Path(
                    self.weights[0] if isinstance(self.weights, list) else self.weights
                ).stem
                if self.weights is not None
                else ""
            )  # weights
            anno_json = str(
                Path(self.data.get("path", "../coco"))
                / "annotations/instances_val2017.json"
            )  # annotations json
            pred_json = str(self.save_dir / f"{w}_predictions.json")  # predictions json
            print(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
            with open(pred_json, "w") as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements(["pycocotools"])
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, "bbox")
                if is_coco:
                    eval.params.imgIds = [
                        int(Path(x).stem) for x in self.dataloader.dataset.img_files
                    ]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(f"pycocotools unable to run: {e}")

        # Return results
        self.model.float()  # for training
        return (
            (
                *self.metric.results(),
                *(self.loss.cpu() / len(self.dataloader)).tolist(),
            ),
            self.metric.maps,
            t,
        )

    def inference(self, img, targets):
        t1 = time_sync()
        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(self.device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        tp = t2 - t1

        # Run model
        out, train_out = self.model(
            img, augment=self.augment
        )  # inference and training outputs
        ti = time_sync() - t2

        # Compute loss
        if self.compute_loss:
            self.loss += self.compute_loss([x.float() for x in train_out], targets)[
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
        tn = time_sync() - t3
        return tp, ti, tn, out

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

    def compute_stat(self, si, img, predn, targets, shape, ratio_pad):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(predn) == 0:
            if nl:
                return (
                    torch.zeros(0, self.niou, dtype=torch.bool),
                    torch.Tensor(),
                    torch.Tensor(),
                    tcls,
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
        return (
            correct.cpu(),
            predn[:, 4].cpu(),
            predn[:, 5].cpu(),
            tcls,
        )  # (correct, conf, pcls, tcls)

    def print(self, seen, nt, nc, names, stats, time):
        # Print results
        pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
        print(
            pf
            % (
                "all",
                seen,
                nt.sum(),
                self.metric.mp,
                self.metric.mr,
                self.metric.map50,
                map,
            )
        )

        # Print results per class
        if (self.verbose or (nc < 50 and not self.training)) and nc > 1 and len(stats):
            for i, c in enumerate(self.metric.ap_class_index):
                print(
                    pf
                    % (
                        names[c],
                        seen,
                        nt[c],
                        self.metric.p[i],
                        self.metric.r[i],
                        self.metric.ap50[i],
                        self.metric.ap[i],
                    )
                )

        # Print speeds
        if not self.training:
            shape = (self.batch_size, 3, self.imgsz, self.imgsz)
            print(
                f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}"
                % time
            )

        if not self.training:
            s = (
                f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}"
                if self.save_txt
                else ""
            )
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

    def eval(self):
        self.device = select_device(self.device, batch_size=self.batch_size)

        # Directories
        self.save_dir = increment_path(
            Path(self.project) / self.name, exist_ok=self.exist_ok
        )  # increment run
        (self.save_dir / "labels" if self.save_txt else self.save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        check_suffix(self.weights, ".pt")
        self.model = attempt_load(
            self.weights, map_location=self.device
        )  # load FP32 model
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(self.imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        self.data = check_dataset(self.data)  # check
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once
        pad = 0.0 if self.task == "speed" else 0.5
        self.task = (
            self.task if self.task in ("train", "val", "test") else "val"
        )  # path to train/val/test images
        self.dataloader = create_dataloader(
            self.data[self.task],
            self.imgsz,
            self.batch_size,
            self.gs,
            self.single_cls,
            pad=pad,
            rect=True,
            prefix=colorstr(f"{self.task}: "),
        )[0]

    def eval_training(self):
        self.device = next(self.model.parameters()).device  # get model device


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

    def set(self, p, r, all_ap, ap_class_index):
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.ap_class_index = ap_class_index
