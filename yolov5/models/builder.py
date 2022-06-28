from lqcv import Registry, build_from_config

BACKBONES = Registry("BACKBONES")
NECKS = Registry("NECKS")
HEADS = Registry("HEADS")
DETECTORS = Registry("DETECTORS")
LOSSES = Registry("LOSSES")


def build_backbone(cfg):
    return build_from_config(cfg, BACKBONES)


def build_neck(cfg):
    return build_from_config(cfg, NECKS)


def build_head(cfg):
    return build_from_config(cfg, HEADS)


def build_detector(cfg):
    return build_from_config(cfg, DETECTORS)

def build_loss(cfg):
    return build_from_config(cfg, LOSSES)
