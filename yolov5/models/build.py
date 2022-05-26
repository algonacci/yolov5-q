import os.path as osp
import torch
from loguru import logger
from .yolo import Model
from ..utils.torch_utils import intersect_dicts


def build_model(cfg, num_class):
    ckpt = None
    if osp.isfile(cfg.PREREAIN):
        ckpt = torch.load(cfg.PREREAIN)  # load checkpoint
        model = Model(
            cfg.PATH or ckpt["model"].yaml,
            ch=3,
            nc=num_class,
        )  # create
        exclude = ["anchor"] if cfg.PATH else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        logger.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {cfg.PREREAIN}"
        )  # report
        del csd
    else:
        model = Model(cfg.PATH, ch=3, nc=num_class)  # create

    return model, ckpt
