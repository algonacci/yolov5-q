# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
My modified version of yolov5 training code, with wandb and evolve, cause I don't need.
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

FILE = Path(__file__).resolve()

from yolov5.utils.general import (
    increment_path,
    get_latest_run,
    print_args,
    set_logging,
)
from yolov5.utils.checker import (
    check_git_status,
    check_requirements,
    check_file,
    check_yaml,
)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.callbacks import Callbacks
from yolov5.core.trainer import Trainer

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weights", type=str, default="yolov5s.pt", help="initial weights path"
    )
    parser.add_argument("-c", "--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("-d", "--data", type=str, default="data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--hyp",
        type=str,
        default="data/hyps/hyp.scratch.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="train, val image size (pixels)",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable autoanchor check")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help='--cache images in "ram" (default) or "disk"',
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument("--adam", action="store_true", help="use torch.optim.Adam() optimizer")
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="EarlyStopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of layers to freeze. backbone=10, all=24",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (disabled if < 1)",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")

    # Weights & Biases arguments
    parser.add_argument("--entity", default=None, help="W&B: Entity")
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="W&B: Upload dataset as artifact table",
    )
    parser.add_argument(
        "--bbox_interval",
        type=int,
        default=-1,
        help="W&B: Set bounding-box image logging interval",
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default="latest",
        help="W&B: Version of dataset artifact to use",
    )
    # additional features
    parser.add_argument("--neg-dir", type=str, default="", help="negative dir")
    parser.add_argument("--bg-dir", type=str, default="", help="background dir")
    parser.add_argument(
        "--area-thr",
        nargs="+",
        type=float,
        default=0.2,
        help="box after augment / origin box areas",
    )
    parser.add_argument(
        "--no-aug-epochs",
        type=int,
        default=15,
        help="box after augment / origin box areas",
    )

    parser.add_argument(
        "-m",
        "--mask",
        action="store_true",
        help="Whether to train the instance segmentation",
    )

    parser.add_argument(
        "-mr",
        "--mask-ratio",
        type=int,
        default=1,
        help="Downsample ratio of the masks gt.",
    )

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=["thop"])

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run()
        )  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "opt.yaml", errors="ignore") as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = "", ckpt, True  # reinstate
        LOGGER.info(f"Resuming training from {ckpt}")
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        assert (
            opt.batch_size % WORLD_SIZE == 0
        ), "--batch-size must be multiple of CUDA device count"
        assert not opt.image_weights, "--image-weights argument is not compatible with DDP training"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    trainer = Trainer(opt.hyp, opt, device, callbacks)
    trainer.train()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info("Destroying process group... ")
        dist.destroy_process_group()


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
