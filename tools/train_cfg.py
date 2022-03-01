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
from omegaconf import OmegaConf

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
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="", help="config file of training")

    opt, unparsed = parser.parse_known_args()
    return opt.file, unparsed


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
        assert len(opt.cfg) or len(
            opt.weights
        ), "either --cfg or --weights must be specified"
        opt.save_dir = str(
            increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
        )

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert (
            torch.cuda.device_count() > LOCAL_RANK
        ), "insufficient CUDA devices for DDP command"
        assert (
            opt.batch_size % WORLD_SIZE == 0
        ), "--batch-size must be multiple of CUDA device count"
        assert (
            not opt.image_weights
        ), "--image-weights argument is not compatible with DDP training"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    trainer = Trainer(
        opt.hyp, opt, device, callbacks, LOGGER, RANK, LOCAL_RANK, WORLD_SIZE
    )
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
    file, unparsed = parse_opt()
    opt = OmegaConf.load(file)
    # opt = OmegaConf.to_yaml(opt)
    # print(unparsed)
    assert len(unparsed) % 2 == 0
    for i in range(0, len(unparsed), 2):
        k = unparsed[i][2:]
        v = unparsed[i + 1]
        try:
            v = eval(v)
        except:
            pass
        OmegaConf.update(opt, k, v)
    main(opt)
