from omegaconf import OmegaConf
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
from lqcv.utils.general import colorstr
from loguru import logger
import os.path as osp
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
from ..models.yolo import Model
from ..utils.autoanchor import check_anchors
from ..data import create_dataloader
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

        self.batch_size = cfg.SOLVER.BATCH_SIZE_PER_GPU
        self.nbs = cfg.SOLVER.NORMAL_BATCH_SIZE  # nominal batch size
        self.accumulate = max(
            round(self.nbs / self.batch_size), 1
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
        # Model
        logger.info(
            colorstr("Creating Model: ") + f"{self.model_cfg or self.weights}..."
        )
        model, ckpt = self.build_model()

        # Optimizer
        self.optimizer, self.scheduler = self.set_optimizer(model)

        # EMA
        self.ema = ModelEMA(self.model) if self.rank == 0 else None

    def build_model(self):
        ckpt = None
        if osp.isfile(self.weights):
            ckpt = torch.load(self.weights)  # load checkpoint
            self.model = Model(
                self.model_cfg or ckpt["model"].yaml,
                ch=3,
                nc=self.num_class,
            )  # create
            exclude = ["anchor"] if self.cfg else []  # exclude keys
            csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            logger.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from {self.weights}"
            )  # report
            del csd
        else:
            model = Model(self.model_cfg, ch=3, nc=self.num_class)  # create

        return model, ckpt

    def set_optimizer(self, cfg, model):
        weight_decay = cfg.WEIGHT_DECAY * (
            self.batch_size * self.accumulate / self.nbs
        )  # scale weight_decay
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

        optimizer = SGD(
            g0, lr=cfg.BASE_LR, momentum=cfg.MOMENTUM, nesterov=True
        )

        optimizer.add_param_group(
            {"params": g1, "weight_decay": weight_decay}
        )  # add g1 with weight_decay
        optimizer.add_param_group({"params": g2})  # add g2 (biases)
        logger.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
            f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias"
        )
        del g0, g1, g2

        # Scheduler
        if self.opt.linear_lr:
            self.lf = (
                lambda x: (1 - x / (self.epochs - 1)) * (1.0 - self.hyp["lrf"])
                + self.hyp["lrf"]
            )  # linear
        else:
            self.lf = one_cycle(1, self.hyp["lrf"], self.epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)
        # plot_lr_scheduler(optimizer, scheduler, self.epochs)
        return optimizer, scheduler
