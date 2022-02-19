import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, create_dataloader_ori
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class Trainer:
    def __init__(self, hyp, opt, device) -> None:
        self.opt = opt
        self.save_dir = opt.save_dir
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.weights = opt.weights
        self.single_cls = opt.single_cls
        self.evolve = opt.evolve
        self.data = opt.data
        self.cfg = opt.cfg
        self.resume = opt.resume
        self.noval = opt.noval
        self.nosave = opt.nosave
        self.workers = opt.workers
        self.freeze = opt.freeze
        self.no_aug_epochs = opt.no_aug_epochs

        self.cuda = device.type != 'cpu'

    def train(self):
        pass

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            pass

    def train_in_iter(self):
        pass

    def train_one_iter(self):
        pass

    def before_train(self):
        w = self.save_dir / 'weights'  # weights dir
        (w.parent if self.evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

        # Save run settings
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(self.opt), f, sort_keys=False)

        self.data_dict = None

        # Loggers(ignored)

        # Config
        self.plots = not self.evolve  # create plots
        init_seeds(1 + RANK)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = self.data_dict or check_dataset(self.data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if self.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if self.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {self.data}'  # check
        is_coco = self.data.endswith('coco.yaml') and nc == 80  # COCO dataset

        # Model
        check_suffix(self.weights, '.pt')  # check weights
        pretrained = self.weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                self.weights = attempt_download(self.weights)  # download if not found locally
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            self.model = Model(self.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (self.cfg or hyp.get('anchors')) and not self.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {self.weights}')  # report
        else:
            self.model = Model(self.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(self.device)  # create

        # Freeze
        freeze = [f'model.{x}.' for x in range(self.freeze)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= self.batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if self.opt.adam:
            optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                    f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
        del g0, g1, g2

        # Scheduler
        if self.opt.linear_lr:
            lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        self.ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        # Resume
        self.start_epoch, best_fitness = 0, 0.0
        if self.pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.resume:
                assert self.start_epoch > 0, f'{self.weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{self.weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

        # Image sizes
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz = check_img_size(self.opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # DP mode
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                            'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        self.train_loader, self.dataset = create_dataloader(train_path, imgsz, self.batch_size // WORLD_SIZE, gs, self.single_cls,
                                                  hyp=hyp, augment=True, cache=self.opt.cache, rect=self.opt.rect, rank=LOCAL_RANK,
                                                  workers=self.workers, image_weights=self.opt.image_weights, quad=self.opt.quad,
                                                  prefix=colorstr('train: '), 
                                                  shuffle=True,
                                                  neg_dir=self.opt.neg_dir,
                                                  bg_dir=self.opt.bg_dir,
                                                  area_thr=self.opt.area_thr)

        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {self.data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            self.val_loader = create_dataloader_ori(val_path, imgsz, self.batch_size // WORLD_SIZE * 2, gs, self.single_cls,
                                           hyp=hyp, cache=None if self.noval else self.opt.cache, rect=True, rank=-1,
                                           workers=self.workers, pad=0.5,
                                           prefix=colorstr('val: '))[0]

            if not self.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if self.plots:
                    plot_labels(labels, names, self.save_dir)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

            # callbacks.run('on_pretrain_routine_end')

        # DDP mode
        if self.cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        # Model parameters
        hyp['box'] *= 3. / self.nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / self.nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / self.nl  # scale to image size and layers
        hyp['label_smoothing'] = self.opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(self.dataset.labels, nc).to(self.device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * self.nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        self.maps = np.zeros(nc)  # mAP per class
        self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.stopper = EarlyStopping(patience=self.opt.patience)
        self.compute_loss = ComputeLoss(model)  # init loss class
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {self.train_loader.num_workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        self.plot_idx = [0, 1, 2]
        if self.no_aug_epochs > 0:
            base_idx = (self.epochs - self.no_aug_epochs) * self.nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

    def after_train(self):
        pass

    def before_epoch(self):
        self.model.train()
        if self.epoch >=(self.epochs - self.no_aug_epochs) :
            self.train_loader.close_augment()
            # dataset.augment = False

        # Update image weights (optional, single-GPU only)
        if self.opt.image_weights:
            cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc  # class weights
            iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
            self.dataset.indices = random.choices(range(self.dataset.n), weights=iw, k=self.dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        self.mloss = torch.zeros(3, device=self.device)  # mean losses
        if RANK != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        pbar = enumerate(self.train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass

    def progress_in_iter(self):
        pass

    def resume_train(self):
        pass

    def evaluate_and_save_model(self):
        pass

    def save_ckpt(self):
        pass
