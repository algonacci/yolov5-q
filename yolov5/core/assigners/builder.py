from lqcv import Registry, build_from_config

ASSIGNER = Registry("ASSIGNER")

def build_assigner(cfg):
    return build_from_config(cfg, ASSIGNER)


