from .datasets import (
    LoadImagesAndLabels,
    LoadImagesAndLabelsAndKeypoints,
    LoadImagesAndLabelsAndMasks,
)
from ..utils.torch_utils import torch_distributed_zero_first
from ..utils.general import colorstr
from ..utils.dist import get_world_size
from .samplers import YoloBatchSampler
from .dataloadering import InfiniteDataLoader, DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import distributed
import os


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    neg_dir="",
    bg_dir="",
    area_thr=0.2,
    mask_head=False,
    mask_downsample_ratio=1,
    keypoint=False,
):
    if rect and shuffle:
        print(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False
    # data_load = LoadImagesAndLabelsAndMasks if mask_head else LoadImagesAndLabels
    if mask_head:
        data_load = LoadImagesAndLabelsAndMasks
    elif keypoint:
        data_load = LoadImagesAndLabelsAndKeypoints
    else:
        data_load = LoadImagesAndLabels

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = data_load(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            neg_dir=neg_dir,
            bg_dir=bg_dir,
            area_thr=area_thr,
        )
        if mask_head:
            dataset.downsample_ratio = mask_downsample_ratio

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    # sampler = InfiniteSampler(len(dataset), seed=0)
    sampler = (
        distributed.DistributedSampler(dataset, shuffle=shuffle)
        if rank != -1
        else RandomSampler(dataset)
    )

    batch_sampler = (
        YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            augment=augment,
        )
        if not rect
        else None
    )
    dataloader = DataLoader(
        dataset,
        num_workers=nw,
        batch_size=1
        if batch_sampler is not None
        else batch_size,  # batch-size and batch-sampler is exclusion
        batch_sampler=batch_sampler,
        pin_memory=True,
        collate_fn=data_load.collate_fn4 if quad else data_load.collate_fn,
        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        # but this will make init_seed() not work.
        # worker_init_fn=worker_init_reset_seed,
    )
    return dataloader, dataset


def build_datasets(cfg, path, stride, rank, mode="train"):
    assert mode in ["train", "val"]
    # data_load = LoadImagesAndLabelsAndMasks if mask_head else LoadImagesAndLabels
    if cfg.MODEL.MASK_ON:
        data_load = LoadImagesAndLabelsAndMasks
    elif cfg.MODEL.KEYPOINT_ON:
        data_load = LoadImagesAndLabelsAndKeypoints
    else:
        data_load = LoadImagesAndLabels

    args = {
        "path": path,
        "img_size": cfg.DATA.IMG_SIZE,
        "batch_size": cfg.SOLVER.BATCH_SIZE_PER_GPU,
        "augment": True if mode == "train" else False,
        "hyp": cfg.DATA.TRANSFORM,
        "rect": False if mode == "train" else True,
        "image_weights": cfg.DATA.IMAGE_WEIGHTS,
        "single_cls": cfg.MODEL.SINGLE_CLS,
        "stride": stride,  # TODO
        "pad": 0.0 if mode == "train" else 0.5,
        "prefix": colorstr(f"{mode}: "),
        "neg_dir": cfg.DATA.NEG_DIR,
        "bg_dir": cfg.DATA.BG_DIR,
    }

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = data_load(**args)
        if cfg.MODEL.MASK_ON:
            dataset.downsample_ratio = cfg.MODEL.MASK_RATIO
    return dataset


def build_dataloader(dataset, is_dist, batch_size, workers):
    sampler = (
        distributed.DistributedSampler(dataset, shuffle=True)
        if is_dist
        else RandomSampler(dataset)
    )
    nw = min(
        [
            os.cpu_count() // get_world_size(),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    batch_sampler = (
        YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            augment=dataset.augment,
        )
        if not dataset.rect
        else None
    )
    dataloader = DataLoader(
        dataset,
        num_workers=nw,
        batch_size=1
        if batch_sampler is not None
        else batch_size,  # batch-size and batch-sampler is exclusion
        batch_sampler=batch_sampler,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        # but this will make init_seed() not work.
        # worker_init_fn=worker_init_reset_seed,
    )
    return dataloader
