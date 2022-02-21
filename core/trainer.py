import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from functools import partial

import numpy as np
import torch
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
from utils.datasets import create_dataloader
from utils.general import (
    labels_to_class_weights,
    labels_to_image_weights,
    init_seeds,
    strip_optimizer,
    check_dataset,
    check_img_size,
    check_suffix,
    one_cycle,
    colorstr,
    methods,
)
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    intersect_dicts,
    torch_distributed_zero_first,
)
from utils.metrics import fitness
from utils.newloggers import NewLoggers

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class Trainer:
    def __init__(self, hyp, opt, device, callbacks) -> None:
        self.hyp = hyp
        self.opt = opt
        self.save_dir = Path(opt.save_dir)
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.weights = opt.weights
        self.single_cls = opt.single_cls
        self.data = opt.data
        self.cfg = opt.cfg
        self.resume = opt.resume
        self.noval = opt.noval
        self.nosave = opt.nosave
        self.workers = opt.workers
        self.freeze = opt.freeze
        self.no_aug_epochs = opt.no_aug_epochs

        self.cuda = device.type != "cpu"
        self.callbacks = callbacks
        self.device = device

        self._initializtion()

    def train(self):
        self.before_train()
        # try:
        #     self.train_in_epoch()
        # except Exception:
        #     raise
        # finally:
        #     self.after_train()
        self.train_in_epoch()
        self.after_epoch()

    def train_in_epoch(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.before_epoch()
            self.train_in_iter()
            early_stop = self.after_epoch()
            if early_stop:
                break

    def train_in_iter(self):
        for i, (imgs, targets, paths, _) in self.pbar:
            imgs = self.before_iter(imgs)
            self.train_one_iter(i, imgs, targets)
            self.after_iter(i, imgs, targets, paths)

    def train_one_iter(self, i, imgs, targets):
        self.ni = (
            i + self.nb * self.epoch
        )  # number integrated batches (since train start)

        # Warmup
        if self.ni <= self.nw:
            self._warmup()

        # Multi-scale
        if self.opt.multi_scale:
            imgs = self._multi_scale(imgs)

        # Forward
        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)  # forward
            loss, self.loss_items = self.compute_loss(
                pred, targets.to(self.device)
            )  # loss scaled by batch_size
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if self.opt.quad:
                loss *= 4.0

        # Backward
        self.scaler.scale(loss).backward()

        # Optimize
        if self.ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = self.ni

    def before_train(self):
        w = self.save_dir / "weights"  # weights dir
        w.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = w / "last.pt", w / "best.pt"

        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp, errors="ignore") as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(
            colorstr("hyperparameters: ")
            + ", ".join(f"{k}={v}" for k, v in hyp.items())
        )

        # Save run settings
        with open(self.save_dir / "hyp.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(self.save_dir / "opt.yaml", "w") as f:
            yaml.safe_dump(vars(self.opt), f, sort_keys=False)

        # Loggers(ignored)
        self.set_logger()

        # Config
        init_seeds(1 + RANK)
        nc, names = self._parse_data()

        # Model
        check_suffix(self.weights, ".pt")  # check weights
        self.pretrained = self.weights.endswith(".pt")
        ckpt = self.load_model(nc, hyp)

        # Optimizer
        self.optimizer, self.scheduler = self.set_optimizer(hyp)

        # EMA
        self.ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        # Resume
        if self.pretrained:
            self.resume_train(ckpt)

        # Image sizes
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.nl = self.model.model[
            -1
        ].nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz = check_img_size(
            self.opt.imgsz, self.gs, floor=self.gs * 2
        )  # verify imgsz is gs-multiple

        # Update self.hyp
        self.hyp = hyp
        
        # initialize dataloader
        self._initialize_loader()

        # DP mode
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning(
                "DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n"
                "See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started."
            )
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(
                self.device
            )
            LOGGER.info("Using SyncBatchNorm()")

        # Trainloader
        self.train_loader, self.dataset = self.create_dataloader(
            path=self.data_dict["train"],
            batch_size=self.batch_size // WORLD_SIZE,
            augment=True,
            cache=self.opt.cache,
            rect=self.opt.rect,
            rank=LOCAL_RANK,
            image_weights=self.opt.image_weights,
            quad=self.opt.quad,
            prefix=colorstr("train: "),
            shuffle=True,
            neg_dir=self.opt.neg_dir,
            bg_dir=self.opt.bg_dir,
            area_thr=self.opt.area_thr,
        )

        self.nb = len(self.train_loader)  # number of batches
        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        assert (
            mlc < nc
        ), f"Label class {mlc} exceeds nc={nc} in {self.data}. Possible class labels are 0-{nc - 1}"

        # Process 0
        if RANK in [-1, 0]:
            self.val_loader = self.create_dataloader(
                path=self.data_dict["val"],
                batch_size=self.batch_size // WORLD_SIZE * 2,
                cache=None if self.noval else self.opt.cache,
                rect=True,
                rank=-1,
                pad=0.5,
                prefix=colorstr("val: "),
            )[0]

            if not self.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if self.plots:
                    plot_labels(labels, names, self.save_dir)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(
                        self.dataset,
                        model=self.model,
                        thr=hyp["anchor_t"],
                        imgsz=self.imgsz,
                    )
                self.model.half().float()  # pre-reduce anchor precision

        # DDP mode
        if self.cuda and RANK != -1:
            self.model = DDP(
                self.model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
            )

        # Model parameters
        self.set_parameters(hyp, nc, names)

        # Start training
        self.nw = max(
            round(hyp["warmup_epochs"] * self.nb), 1000
        )  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training

        self.maps = np.zeros(nc)  # mAP per class
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.stopper = EarlyStopping(patience=self.opt.patience)
        self.compute_loss = ComputeLoss(self.model)  # init loss class

        LOGGER.info(
            f"Image sizes {self.imgsz} train, {self.imgsz} val\n"
            f"Using {self.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs} epochs..."
        )
        if self.no_aug_epochs > 0:
            base_idx = (self.epochs - self.no_aug_epochs) * self.nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        # initialize eval
        self._initialize_eval()

    def after_train(self):
        if RANK not in [-1, 0]:
            return
        LOGGER.info(
            f"\n{self.epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.t0) / 3600:.3f} hours."
        )
        for f in self.last, self.best:
            if not f.exists():
                continue
            strip_optimizer(f)  # strip optimizers
            if f is not self.best:
                continue
            LOGGER.info(f"\nValidating {f}...")
            self.results, _, _ = self.eval(
                model=attempt_load(f, self.device).half(),
                iou_thres=0.65
                if self.is_coco
                else 0.60,  # best pycocotools results at 0.65
                save_json=self.is_coco,
                verbose=True,
                plots=True,
            )  # val best model with plots
            if self.is_coco:
                self.callbacks.run(
                    "on_fit_epoch_end",
                    list(self.mloss) + list(self.results) + self.lr,
                    self.epoch,
                    self.best_fitness,
                    self.fi,
                )

        self.callbacks.run("on_train_end", self.plots, self.epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        torch.cuda.empty_cache()
        return self.results

    def before_epoch(self):
        self.model.train()
        if self.epoch >= (self.epochs - self.no_aug_epochs):
            self.train_loader.close_augment()

        # Update image weights (optional, single-GPU only)
        if self.opt.image_weights:
            cw = (
                self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc
            )  # class weights
            iw = labels_to_image_weights(
                self.dataset.labels, nc=self.nc, class_weights=cw
            )  # image weights
            self.dataset.indices = random.choices(
                range(self.dataset.n), weights=iw, k=self.dataset.n
            )  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        self.mloss = torch.zeros(3, device=self.device)  # mean losses
        if RANK != -1:
            self.train_loader.batch_sampler.sampler.set_epoch(self.epoch)

        self.pbar = enumerate(self.train_loader)
        LOGGER.info(
            ("\n" + "%10s" * 7)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size")
        )
        if RANK in [-1, 0]:
            self.pbar = tqdm(self.pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()

    def after_epoch(self):
        # Scheduler
        self.scheduler.step()

        if RANK not in [-1, 0]:
            return
        # mAP
        self.ema.update_attr(
            self.model,
            include=["yaml", "nc", "hyp", "names", "stride", "class_weights"],
        )

        fi = self.evaluate_and_save_model()

        # Stop Single-GPU
        if RANK == -1 and self.stopper(epoch=self.epoch, fitness=fi):
            return True

    def before_iter(self, imgs):
        imgs = (
            imgs.to(self.device, non_blocking=True).float() / 255.0
        )  # uint8 to float32, 0-255 to 0.0-1.0
        return imgs

    def after_iter(self, i, imgs, targets, paths):
        # Log
        if RANK not in [-1, 0]:
            return
        self.mloss = (self.mloss * i + self.loss_items) / (i + 1)  # update mean losses
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        self.pbar.set_description(
            ("%10s" * 2 + "%10.4g" * 5)
            % (
                f"{self.epoch}/{self.epochs - 1}",
                mem,
                *self.mloss,
                targets.shape[0],
                imgs.shape[-1],
            )
        )
        self.callbacks.run(
            "on_train_batch_end",
            self.ni,
            self.model,
            imgs,
            targets,
            paths,
            self.plots,
            self.opt.sync_bn,
            self.plot_idx,
        )

    def progress_in_iter(self):
        pass

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
        if self.resume:
            assert (
                self.start_epoch > 0
            ), f"{self.weights} training to {self.epochs} epochs is finished, nothing to resume."
        if self.epochs < self.start_epoch:
            LOGGER.info(
                f"{self.weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt

    def evaluate_and_save_model(self):
        lr = [x["lr"] for x in self.optimizer.param_groups]  # for loggers
        final_epoch = (self.epoch + 1 == self.epochs) or self.stopper.possible_stop
        if not self.noval or final_epoch:  # Calculate mAP
            self.results, self.maps, _ = self.eval(
                model=self.ema.ema,
                plots=False,
            )

        # Update best mAP
        fi = fitness(
            np.array(self.results).reshape(1, -1)
        )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > self.best_fitness:
            self.best_fitness = fi
        log_vals = list(self.mloss) + list(self.results) + lr
        self.callbacks.run(
            "on_fit_epoch_end", log_vals, self.epoch, self.best_fitness, fi
        )

        # Save model
        if (not self.nosave) or final_epoch:  # if save
            self.save_ckpt(fi)
            self.callbacks.run(
                "on_model_save",
                self.last,
                self.epoch,
                final_epoch,
                self.best_fitness,
                fi,
            )
        return fi

    def save_ckpt(self, fi):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
        }

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == fi:
            torch.save(ckpt, self.best)
        if (
            (self.epoch > 0)
            and (self.opt.save_period > 0)
            and (self.epoch % self.opt.save_period == 0)
        ):
            torch.save(ckpt, self.w / f"epoch{self.epoch}.pt")
        del ckpt

    def load_model(self, nc, hyp):
        ckpt = None
        if self.pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                self.weights = attempt_download(
                    self.weights
                )  # download if not found locally
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            self.model = Model(
                self.cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
            ).to(
                self.device
            )  # create
            exclude = (
                ["anchor"]
                if (self.cfg or hyp.get("anchors")) and not self.resume
                else []
            )  # exclude keys
            csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(
                csd, self.model.state_dict(), exclude=exclude
            )  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from {self.weights}"
            )  # report
        else:
            self.model = Model(self.cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(
                self.device
            )  # create

        del csd

        # Freeze
        freeze = [f"model.{x}." for x in range(self.freeze)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False

        return ckpt

    def set_optimizer(self, hyp):
        self.nbs = 64  # nominal batch size
        self.accumulate = max(
            round(self.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        hyp["weight_decay"] *= (
            self.batch_size * self.accumulate / self.nbs
        )  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(
                v.weight, nn.Parameter
            ):  # weight (with decay)
                g1.append(v.weight)

        if self.opt.adam:
            optimizer = Adam(
                g0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
            )  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group(
            {"params": g1, "weight_decay": hyp["weight_decay"]}
        )  # add g1 with weight_decay
        optimizer.add_param_group({"params": g2})  # add g2 (biases)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
            f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias"
        )
        del g0, g1, g2

        # Scheduler
        if self.opt.linear_lr:
            self.lf = (
                lambda x: (1 - x / (self.epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]
            )  # linear
        else:
            self.lf = one_cycle(1, hyp["lrf"], self.epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.lf
        )  # plot_lr_scheduler(optimizer, scheduler, epochs)
        return optimizer, scheduler

    def set_logger(self):
        if RANK not in [-1, 0]:
            return
        loggers = NewLoggers(
            save_dir=self.save_dir, opt=self.opt, logger=LOGGER
        )  # loggers instance

        # Register actions
        for k in methods(loggers):
            self.callbacks.register_action(k, callback=getattr(loggers, k))

    def set_parameters(self, hyp, nc, names):
        hyp["box"] *= 3.0 / self.nl  # scale to layers
        hyp["cls"] *= nc / 80.0 * 3.0 / self.nl  # scale to classes and layers
        hyp["obj"] *= (
            (self.imgsz / 640) ** 2 * 3.0 / self.nl
        )  # scale to image size and layers
        hyp["label_smoothing"] = self.opt.label_smoothing
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.class_weights = (
            labels_to_class_weights(self.dataset.labels, nc).to(self.device) * nc
        )  # attach class weights
        self.model.names = names

    def _initializtion(self):
        self.start_epoch, self.best_fitness = 0, 0.0
        self.last_opt_step = -1
        self.results = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.plot_idx = [0, 1, 2]
        self.plots = True  # create plots
        self.t0 = time.time()


    def _parse_data(self):
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = check_dataset(self.data)  # check if None
        nc = 1 if self.single_cls else int(data_dict["nc"])  # number of classes
        names = (
            ["item"]
            if self.single_cls and len(data_dict["names"]) != 1
            else data_dict["names"]
        )  # class names
        assert (
            len(names) == nc
        ), f"{len(names)} names found for nc={nc} dataset in {self.data}"  # check
        self.is_coco = self.data.endswith("coco.yaml") and nc == 80  # COCO dataset
        self.data_dict = data_dict
        return nc, names

    def _warmup(self):
        xi = [0, self.nw]  # x interp
        # compute_loss.gr = np.interp(self.ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
        self.accumulate = max(
            1, np.interp(self.ni, xi, [1, self.nbs / self.batch_size]).round()
        )
        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(
                self.ni,
                xi,
                [
                    self.hyp["warmup_bias_lr"] if j == 2 else 0.0,
                    x["initial_lr"] * self.lf(self.epoch),
                ],
            )
            if "momentum" in x:
                x["momentum"] = np.interp(
                    self.ni, xi, [self.hyp["warmup_momentum"], self.hyp["momentum"]]
                )

    def _multi_scale(self, imgs):
        sz = (
            random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs)
            // self.gs
            * self.gs
        )  # size
        sf = sz / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [
                math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]
            ]  # new shape (stretched to gs-multiple)
            imgs = nn.functional.interpolate(
                imgs, size=ns, mode="bilinear", align_corners=False
            )
        return imgs

    def _initialize_loader(self):
        # functions
        self.create_dataloader = partial(
            create_dataloader,
            imgsz=self.imgsz,
            stride=self.gs,
            single_cls=self.single_cls,
            hyp=self.hyp,
            workers=self.workers,
        )

    def _initialize_eval(self):
        self.eval = partial(
            val.run,
            data=self.data_dict,
            batch_size=self.batch_size // WORLD_SIZE * 2,
            imgsz=self.imgsz,
            single_cls=self.single_cls,
            save_dir=self.save_dir,
            dataloader=self.val_loader,
            callbacks=self.callbacks,
            compute_loss=self.compute_loss,
        )
