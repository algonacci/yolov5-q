from .yolov5 import Yolov5Exp
from ..data.datasets import create_dataloader, LoadImagesAndLabels
from ..utils.general import colorstr, methods
from ..utils.newloggers import NewLoggersMask
from ..utils.callbacks import Callbacks
from yolov5.core.evaluator import Yolov5Evaluator


class Yolov5SegExp(Yolov5Exp):
    def __init__(self, opt):
        super().__init__(opt)
        # initialize metric
        self.results = [0 for _ in range(12)]
        self.plots = False  # create plots
        self.mask_ratio = opt.mask_ratio

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
            mask_head=True,
            mask_downsample_ratio=self.mask_ratio,
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
            mask_head=True,
            mask_downsample_ratio=self.mask_ratio,
        )[0]
        return val_loader

    def get_evaluator(self):
        evaluator = Yolov5Evaluator(
            data=self.data,
            single_cls=self.single_cls,
            save_dir=self.save_dir,
            plots=True,
            verbose=False,
            mask=True,
            mask_downsample_ratio=self.mask_ratio,
        )
        return evaluator

    def set_logger(self, rank, save_dir):
        if rank not in [-1, 0]:
            return
        callbacks = Callbacks()
        loggers = NewLoggersMask(
            save_dir=save_dir,
            rank=rank,
        )  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        return callbacks
