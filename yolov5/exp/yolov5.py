from .base import BaseExp
from ..data.datasets import create_dataloader, LoadImagesAndLabels
from ..utils.general import colorstr, methods
from ..models.loss import ComputeLoss
from ..utils.checker import check_dataset
from ..utils.newloggers import NewLoggers
from ..utils.callbacks import Callbacks
from yolov5.core.evaluator import Yolov5Evaluator
from omegaconf import OmegaConf
from pathlib import Path


class Yolov5Exp(BaseExp):
    def __init__(self, opt):
        super().__init__()
        # initialize metric
        self.results = [0 for _ in range(7)]
        self.plots = True  # create plots

        # dataloader
        self.data = opt.data
        self.data_dict = check_dataset(self.data)  # check if None
        self.single_cls = opt.single_cls
        self.workers = opt.workers
        self.rect = opt.rect
        self.image_weights = opt.image_weights
        self.quad = opt.quad
        self.cache = opt.cache
        self.neg_dir = opt.neg_dir
        self.bg_dir = opt.bg_dir
        self.area_thr = opt.area_thr

        self.save_dir = Path(opt.save_dir)

        if isinstance(self.hyp, str):
            with open(self.hyp, errors="ignore") as f:
                self.hyp = OmegaConf.load(f)  # load hyps dict

    def get_data_loader(self, batch_size, rank, imgsz, stride):
        train_loader, dataset = create_dataloader(
            path=self.data_dict["train"],
            augment=True,
            rect=self.rect,
            imgsz=imgsz,
            stride=stride,
            single_cls=self.single_cls,
            hyp=self.hyp,
            shuffle=True,
            neg_dir=self.neg_dir,
            bg_dir=self.bg_dir,
            area_thr=self.area_thr,
            cache=self.cache,
            image_weights=self.image_weights,
            prefix=colorstr("train: "),
            batch_size=batch_size,
            workers=self.workers,
            quad=self.quad,
            rank=rank,
        )
        return train_loader, dataset

    def get_eval_loader(self, batch_size, imgsz, stride, noval=False):
        val_loader = create_dataloader(
            path=self.data_dict["val"],
            imgsz=imgsz,
            stride=stride,
            single_cls=self.single_cls,
            hyp=self.hyp,
            workers=self.workers,
            batch_size=batch_size * 2,
            cache=None if noval else self.cache,
            rect=True,
            rank=-1,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]
        return val_loader

    def get_evaluator(self):
        evaluator = Yolov5Evaluator(
            data=self.data,
            single_cls=self.single_cls,
            save_dir=self.save_dir,
            plots=True,
            verbose=False,
        )
        return evaluator

    def set_logger(self, rank):
        if rank not in [-1, 0]:
            return
        callbacks = Callbacks()
        loggers = NewLoggers(
            save_dir=self.save_dir,
            rank=rank,
        )  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        return callbacks
