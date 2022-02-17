import os
import random
import numpy as np
import torch.nn.functional as F
from utils.general import *
from utils.plots import plot_images_and_masks
from utils.datasets import LoadImagesAndLabelsAndMasks
import cv2
from utils.general import init_seeds

names = ['balloon']
seed = 2
init_seeds(seed)
colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]

with open('data/hyps/hyp.scratch.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

# dataset = LoadImagesAndLabels('data/play_phone1216/images/train', img_size=640, augment=True, cache_images=False,
#                               hyp=hyp)
dataset = LoadImagesAndLabelsAndMasks(
    # '/d/baidubase/COCO/val_yolo/images/train',
    # 'data/license_plates/images/train/',
    # '/d/projects/research/yolov5/data/coco/train2017.txt',
    # '/d/projects/research/yolov5/data/coco/val2017.txt',
    '/home/laughing/yolov5/data/seg/balloon/images/train',
    img_size=640,
    augment=False,
    cache_images=False,
    hyp=hyp,
)
dataset.mosaic = True

save = False
save_dir = '/d/projects/yolov5/data/play_phone0115/show_labels'

if save and not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    sampler=None,
    pin_memory=True,
    collate_fn=LoadImagesAndLabelsAndMasks.collate_fn)
cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)

for i, (imgs, targets, paths, _, masks) in enumerate(dataloader):
    # for i, (imgs, targets, paths, _) in enumerate(dataset):
    #     print(targets)
    print(targets)
    result = plot_images_and_masks(images=imgs,
                          targets=targets,
                          paths=paths,
                          masks=masks)
    cv2.imshow('mosaic', result[:, :, ::-1])
    if cv2.waitKey(0) == ord('q'):  # q to quit
        break
    continue
    # imgs = imgs.numpy().astype(np.uint8).transpose((1, 2, 0))
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    # targets = targets.numpy()
    # temp = targets[:, 2:6].copy()
    # targets[:, 2:6] = xywh2xyxy(temp * 960)
    # for idx, cls, *xyxy in targets:
    #     label = '%s' % (names[int(cls)])
    #     plot_one_box(xyxy, imgs, label=label, color=colors[int(cls)], line_thickness=2)
    # # print(targets[targets[:, 0] == 0])
    # if save and len(targets[targets[:, 1] == 0]) > 0:
    #     cv2.imwrite(os.path.join(save_dir, os.path.split(paths)[-1]), imgs)
    # cv2.imshow('p', imgs)
    # cv2.waitKey(0)
