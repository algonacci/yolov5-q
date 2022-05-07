import os
import random
import numpy as np
import torch.nn.functional as F
from yolov5.utils.general import *
from yolov5.utils.boxes import xywh2xyxy
from yolov5.utils.plots import plot_images_boxes_and_masks, plot_images_keypoints
from yolov5.data import LoadImagesAndLabelsAndMasks, create_dataloader
import cv2
from yolov5.utils.general import init_seeds

names = ["balloon"]
seed = 2
init_seeds(seed)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

with open("data/hyps/hyp.scratch.yaml") as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

save = False
save_dir = "/d/projects/yolov5/data/play_phone0115/show_labels"

if save and not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataloader, dataset = create_dataloader(
            # '/d/projects/research/yolov5/data/coco/train2017.txt',
            # '/home/laughing/code/yolov5-q/data/balloon/images/train',
            '/home/laughing/code/yolov5-q/data/keypoints',
            imgsz=640,
            batch_size=4,
            stride=32,
            augment=True,
            shuffle=True,
            rank=-1,
            hyp=hyp,
            mask_head=False,
            keypoint=True,
            )
cv2.namedWindow("mosaic", cv2.WINDOW_NORMAL)

# mask test
# for i, (imgs, targets, paths, _, masks) in enumerate(dataloader):
#     if getattr(dataset, 'downsample_ratio', 1) != 1 and masks is not None:
#         masks = F.interpolate(
#             masks[None, :],
#             (640, 640),
#             mode="bilinear",
#             align_corners=False,
#         ).squeeze(0)
#
#     result = plot_images_boxes_and_masks(images=imgs, targets=targets, paths=paths, masks=masks)
#     cv2.imshow("mosaic", result[:, :, ::-1])
#     if cv2.waitKey(0) == ord("q"):  # q to quit
#         break
#     continue

# keypoint test
for i, (imgs, targets, paths, _, keypoints) in enumerate(dataloader):
    result = plot_images_keypoints(images=imgs, targets=targets, keypoints=keypoints, paths=paths)
    cv2.imshow("mosaic", result[:, :, ::-1])
    if cv2.waitKey(0) == ord("q"):  # q to quit
        break
