from omegaconf import OmegaConf
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
from lqcv.utils.general import colorstr
from loguru import logger
from copy import deepcopy
import os.path as osp
import random
import math
import os
from ..utils.logger import setup_logger
from ..utils.dist import get_rank, get_world_size

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

from ..models.experimental import attempt_load
from ..models import build_model
from ..utils.autoanchor import check_anchors
from ..data import build_datasets, build_dataloader
from ..utils.general import (
    labels_to_class_weights,
    labels_to_image_weights,
    init_seeds,
    strip_optimizer,
    one_cycle,
    colorstr,
    methods,
)
from ..utils.checker import (
    check_dataset,
    check_img_size,
    check_suffix,
)
from ..utils.downloads import attempt_download
from ..models.loss import ComputeLoss
from ..utils.plots import plot_labels
from ..utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    intersect_dicts,
    torch_distributed_zero_first,
)
from ..utils.metrics import fitness
from ..utils.newloggers import NewLoggers, NewLoggersMask
from .evaluator import Yolov5Evaluator


def build_optimizer(cfg, model, scale=1):
    weight_decay = cfg.WEIGHT_DECAY * scale  # scale weight_decay
    logger.info(f"Scaled weight_decay = {weight_decay}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):  # weight (with decay)
            g1.append(v.weight)

    optimizer = SGD(g0, lr=cfg.BASE_LR, momentum=cfg.MOMENTUM, nesterov=True)

    optimizer.add_param_group(
        {"params": g1, "weight_decay": weight_decay}
    )  # add g1 with weight_decay
    optimizer.add_param_group({"params": g2})  # add g2 (biases)
    logger.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
        f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias"
    )
    del g0, g1, g2
    return optimizer


def build_scheduler(cfg, max_epoch, optimizer):
    # Scheduler
    assert cfg.LR_SCHEDULER in ["linear", "cosine"]
    if cfg.LR_SCHEDULER == "linear":
        lf = lambda x: (1 - x / (max_epoch - 1)) * (1.0 - cfg.LRF) + cfg.LRF  # linear
    else:
        lf = one_cycle(1, cfg.LRF, max_epoch)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, self.epochs)
    return scheduler, lf


class Trainer:
    def __init__(self, cfg, local_rank, nosave=False, noval=False) -> None:
        self.is_distributed = get_world_size() > 1
        self.local_rank = local_rank
        self.rank = get_rank()

        self.cfg = cfg
        self.img_size = cfg.DATASET.IMG_SIZE
        self.model_cfg = cfg.MODEL.PATH
        self.weights = cfg.MODEL.PRETRAIN
        self.single_cls = cfg.MODEL.SINGLE_CLS
        self.no_aug_epochs = cfg.SOLVER.NO_AUG_EPOCH

        self.batch_size = cfg.SOLVER.BATCH_SIZE_PER_GPU
        self.normal_batch_size = cfg.SOLVER.NORMAL_BATCH_SIZE  # nominal batch size
        self.accumulate = max(
            round(self.normal_batch_size / self.batch_size), 1
        )  # accumulate loss before optimizing
        self.data_dict = check_dataset(cfg.DATA.PATH)
        self.num_class = 1 if self.single_cls else int(self.data_dict["nc"])

        self.max_epoch = cfg.SOLVER.NUM_EPOCH
        self.save_dir = cfg.OUTPUT
        self.resume_dir = cfg.get("RESUME_DIR", None)
        self.hyp = cfg.HYP

        self.scaler = amp.GradScaler(enabled=True)

        self.best_fitness = 0
        self.best_epoch = 0
        self.start_epoch = 0
        self.last = osp.join(self.save_dir, "weights", "last.pt")
        self.best = osp.join(self.save_dir, "weights", "best.pt")
        self.nosave = nosave
        self.noval = noval

        self.last_opt_step = -1
        # P(B), R(B), mAP@.5(B), mAP@.5-.95(B),
        # P(M), R(M), mAP@.5(M), mAP@.5-.95(M),
        # val_loss(box, seg, obj, cls)
        self.results = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) if self.mask else (0, 0, 0, 0, 0, 0, 0)
        )
        self.plot_idx = [0, 1, 2]

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def before_train(self):
        os.makedirs(osp.join(self.save_dir, "weights"), exist_ok=True)
        setup_logger(
            self.save_dir,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )
        init_seeds(1 + self.rank)
        # Model
        logger.info(
            colorstr("Creating Model: ") + f"{self.model_cfg or self.weights}..."
        )
        # build model from model cfg or weights
        model, ckpt = build_model(cfg=self.cfg.MODEL, num_class=self.num_class)
        # get cls names
        names = (
            ["item"]
            if self.single_cls and len(self.data_dict["names"]) != 1
            else self.data_dict["names"]
        )  # class names
        # add some attr to model
        model = self.update_attr(model, names)
        # EMA
        self.ema = ModelEMA(model) if self.rank == 0 else None

        # Optimizer
        self.optimizer = build_optimizer(
            self.cfg.SOLVER,
            model,
            scale=(self.batch_size * self.accumulate / self.normal_batch_size),
        )
        self.scheduler, self.lf = build_scheduler(
            self.cfg.SOLVER, self.max_epoch, self.optimizer
        )

        # NOTE: resume epoch, optimizer, ema
        if ckpt is not None:
            self.resume_train(ckpt)
        model.cuda()
        if self.is_distributed:
            model = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        self.model = model
        self.model.train()

        # stride for dataset and multi-scale training
        self.stride = max(int(self.model.stride.max()), 32)
        train_dataset = build_datasets(
            self.cfg,
            self.data_dict["train"],
            stride=self.stride,
            rank=self.rank,
            mode="train",
        )
        self.train_loader = build_dataloader(
            train_dataset, self.is_distributed, self.batch_size, self.cfg.NUM_WORKERS
        )
        self.max_iter = len(self.train_loader)  # number of batches

        # check cls index of labels
        mlc = int(
            np.concatenate(train_dataset.labels, 0)[:, 0].max()
        )  # max label class
        assert mlc < self.num_class, (
            f"Label class {mlc} exceeds nc={self.num_class} in {self.cfg.DATA.PATH}. "
            "Possible class labels are 0-{self.num_class - 1}"
        )

        if self.rank == 0:
            labels = np.concatenate(self.dataset.labels, 0)
            # TODO: nosave
            if not self.nosave:
                plot_labels(labels, names, self.save_dir)

            # Anchors
            check_anchors(
                self.dataset,
                model=self.model,
                thr=self.hyp["anchor_t"],
                imgsz=self.img_size,
            )
            self.model.half().float()  # pre-reduce anchor precision

        # warmup epochs
        self.warmup_iters = max(
            round(self.cfg.DATA.WARMUP_EPOCHS * self.max_iter), 1000
        )  # number of warmup iterations, max(3 epochs, 1k iterations)

        # TODO: loss function
        self.compute_loss = ComputeLoss(self.model)  # init loss class

        if self.no_aug_epochs > 0:
            base_idx = (self.max_epoch - self.no_aug_epochs) * self.max_iter
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        self.logger.info(
            f"Image sizes {self.img_size} train, {self.img_size} val\n"
            f"Using {self.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {colorstr('bold', None if self.nosave else self.save_dir)}\n"
            f"Starting training for {self.max_epoch} epochs..."
        )

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter, (imgs, targets, paths, _, masks) in self.pbar:
            imgs = (
                imgs.to(self.device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if self.progress_in_iter <= self.warmup_iters:
                self._warmup()

            # Multi-scale
            if self.opt.multi_scale:
                imgs = self._multi_scale(imgs)
            # Forward
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(
                    pred,
                    targets.to(self.device),
                    masks=masks.to(self.device) if self.mask else masks,
                )  # loss scaled by batch_size
                if self.is_distributed:
                    loss *= (
                        get_world_size()
                    )  # gradient averaged between devices in DDP mode

            # Backward
            self.scaler.scale(loss).backward()

            # Optimize
            if self.progress_in_iter - self.last_opt_step >= self.accumulate:
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
                self.last_opt_step = self.progress_in_iter

    def before_epoch(self):
        nloss = 4 if self.mask else 3  #  (obj, cls, box, [seg])
        self.mloss = torch.zeros(nloss, device=self.device)  # mean losses

        self.model.train()
        if self.epoch == (self.epochs - self.no_aug_epochs):
            self.logger.info("--->No mosaic aug now!")
            self.train_loader.close_augment()
            if self.rank in [-1, 0]:
                self.save_ckpt(save_file=self.last_mosaic)

        if self.is_distributed:
            self.train_loader.batch_sampler.sampler.set_epoch(self.epoch)

        s = (
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "seg", "obj", "cls", "labels", "img_size")
            if self.mask
            else ("\n" + "%10s" * 7)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size")
        )
        logger.info(s)

        self.pbar = enumerate(self.train_loader)
        if self.rank == 0:
            self.pbar = tqdm(self.pbar, total=self.batches)  # progress bar
        self.optimizer.zero_grad()

    def after_epoch(self):
        """
        - lr scheduler
        - evaluation
        - save model
        """
        # Scheduler
        self.scheduler.step()

        if self.rank != 0:
            return
        # mAP
        self.ema.update_attr(
            self.model,
            include=["yaml", "nc", "hyp", "names", "stride"],
        )

        # TODO
        self.evaluate_and_save_model()

    @property
    def global_iter(self):
        return (
            self.iter + self.max_iter * self.epoch
        )  # number integrated batches (since train start)

    def resume_train(self, ckpt):
        # Optimizer
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_fitness = ckpt["best_fitness"]

        # EMA
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]

        # Epochs
        self.start_epoch = ckpt["epoch"] + 1
        if self.max_epoch < self.start_epoch:
            self.logger.info(
                f"{self.weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.max_epoch += ckpt["epoch"]  # finetune additional epochs
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        del ckpt

    def update_attr(self, model, names):
        nl = model.model[-1].nl
        self.hyp.BOX *= 3.0 / nl  # scale to layers
        self.hyp.CLS *= self.num_class / 80.0 * 3.0 / nl  # scale to classes and layers
        self.hyp.OBJ *= (
            (self.img_size / 640) ** 2 * 3.0 / nl
        )  # scale to image size and layers
        model.nc = self.num_class  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model
        model.names = names
        return model

    def save_ckpt(self, save_file, best=False):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
        }

        # Save last, best and delete
        torch.save(ckpt, save_file)
        if best:
            torch.save(ckpt, self.best)
        del ckpt

    def _warmup(self):
        xi = [0, self.warmup_iters]  # x interp
        self.accumulate = max(
            1,
            np.interp(
                self.global_iter, xi, [1, self.normal_batch_size / self.batch_size]
            ).round(),
        )
        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(
                self.global_iter,
                xi,
                [
                    self.hyp.WARMUP_BIAS_LR if j == 2 else 0.0,
                    x["initial_lr"] * self.lf(self.epoch),
                ],
            )
            if "momentum" in x:
                x["momentum"] = np.interp(
                    self.global_iter, xi, [self.hyp.WARMUP_MOMENTUM, self.hyp.MOMENTUM]
                )

    def _multi_scale(self, imgs):
        sz = (
            random.randrange(self.img_size * 0.5, self.img_size * 1.5 + self.stride)
            // self.stride
            * self.stride
        )  # size
        sf = sz / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [
                math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
            ]  # new shape (stretched to gs-multiple)
            imgs = nn.functional.interpolate(
                imgs, size=ns, mode="bilinear", align_corners=False
            )
        return imgs
