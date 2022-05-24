from omegaconf import OmegaConf
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
from lqcv.utils.general import colorstr
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
    def __init__(self, cfg, local_rank) -> None:
        self.is_distributed = get_world_size() > 1
        self.local_rank = local_rank
        self.rank = get_rank()

        self.cfg = cfg
        self.img_size = cfg.DATASET.IMG_SIZE
        self.model_cfg = cfg.MODEL.PATH
        self.pretrain_weight = cfg.MODEL.PRETRAIN

        self.batch_size = cfg.SOLVER.BATCH_SIZE_PER_GPU
        self.nbs = cfg.SOLVER.NORMAL_BATCH_SIZE  # nominal batch size
        self.accumulate = max(
            round(self.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing

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
        self.logger.info(colorstr("Creating Model: ") + f"{self.backbone}...")
        # Optimizer
        self.optimizer, self.scheduler = self.set_optimizer()

        # EMA
        self.ema = ModelEMA(self.model) if self.rank == 0 else None
