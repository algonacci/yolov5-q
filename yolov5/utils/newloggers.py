# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter

from .general import colorstr
from .plots import (
    plot_images,
    plot_images_and_masks,
    plot_results,
    plot_results_with_masks,
)
from .torch_utils import de_parallel

LOGGERS = ("csv", "tb")  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv("RANK", -1))


class NewLoggers:
    """Loggers without wandb, cause I don't really use `wandb` and `wandb` related codes are noisy."""
    def __init__(
        self,
        save_dir=None,
        opt=None,
        logger=None,
        include=LOGGERS,
    ):
        self.save_dir = save_dir
        self.opt = opt
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # metrics
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = [
            "best/epoch",
            "best/precision",
            "best/recall",
            "best/mAP_0.5",
            "best/mAP_0.5:0.95",
        ]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and s.exists():
            prefix = colorstr("TensorBoard: ")
            self.logger.info(
                f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(s))

    def on_pretrain_routine_end(self):
        pass

    def on_train_batch_end(
        self, ni, model, imgs, targets, masks, paths, plots, sync_bn, plot_idx
    ):
        # Callback runs on train batch end
        if plots and self.save_dir.exists():
            if ni == 0:
                if (
                    not sync_bn
                ):  # tb.add_graph() --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # suppress jit trace warning
                        self.tb.add_graph(
                            torch.jit.trace(
                                de_parallel(model), imgs[0:1], strict=False
                            ),
                            [],
                        )
            if plot_idx is not None and ni in plot_idx:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                Thread(
                    target=plot_images, args=(imgs, targets, paths, f), daemon=True
                ).start()
            # if ni < 3:
            #     f = self.save_dir / f'train_batch{ni}.jpg'  # filename
            #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_end(self):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv and self.save_dir.exists():
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = (
                ""
                if file.exists()
                else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")
            )  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        pass

    def on_train_end(self, plots, epoch, masks=False):
        plts = plot_results_with_masks if masks else plot_results
        # Callback runs on training end
        if plots and self.save_dir.exists():
            plts(file=self.save_dir / "results.csv")  # save results.png
        files = [
            "results.png",
            "confusion_matrix.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [
            (self.save_dir / f) for f in files if (self.save_dir / f).exists()
        ]  # filter

        if self.tb:
            import cv2

            for f in files:
                self.tb.add_image(
                    f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC"
                )

    def on_params_update(self):
        # Update hyperparams or configs of the experiment
        # params: A dict containing {param: value} pairs
        pass


class NewLoggersMask(NewLoggers):
    def __init__(
        self,
        save_dir=None,
        opt=None,
        logger=None,
        include=LOGGERS,
    ):
        super().__init__(save_dir, opt, logger, include)
        self.keys = [
            "train/box_loss",
            "train/seg_loss",  # train loss
            "train/obj_loss",
            "train/cls_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP_0.5(B)",
            "metrics/mAP_0.5:0.95(B)",  # metrics
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP_0.5(M)",
            "metrics/mAP_0.5:0.95(M)",  # metrics
            "val/box_loss",
            "val/seg_loss",  # val loss
            "val/obj_loss",
            "val/cls_loss",
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = [
            "best/epoch",
            "best/precision",
            "best/recall",
            "best/mAP_0.5",
            "best/mAP_0.5:0.95",
        ]

    def on_train_batch_end(
        self, ni, model, imgs, targets, masks, paths, plots, sync_bn, plot_idx
    ):
        # Callback runs on train batch end
        if plots and self.save_dir.exists():
            if ni == 0:
                if (
                    not sync_bn
                ):  # tb.add_graph() --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # suppress jit trace warning
                        self.tb.add_graph(
                            torch.jit.trace(
                                de_parallel(model), imgs[0:1], strict=False
                            ),
                            [],
                        )
            if plot_idx is not None and ni in plot_idx:
                # if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                Thread(
                    target=plot_images_and_masks,
                    args=(imgs, targets, masks, paths, f),
                    daemon=True,
                ).start()

    def on_fit_epoch_end(self, vals, epoch):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv and self.save_dir.exists():
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = (
                ""
                if file.exists()
                else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")
            )  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
