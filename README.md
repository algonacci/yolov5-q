

## TODO
- [X] `plot_results`
- [X] `process_masks` mask cuda out of memory
- [X] `detect_seg.py`
- [X] support flip augmentation
- [X] val
- [X] clean `dataset.py`
- [X] DetectSegment head support `gw`
- [X] smaller gt_masks for saving memory(support train.py only)
- [X] test `scale_coords` influence for map
- [X] nosave
- [ ] `train_cfg.py`
- [ ] support albumentations
- [ ] Mixup
- [ ] DetectSegment head support `gd`
- [ ] better way to compute seg loss
- [ ] DDP
- [ ] coco eval
- [ ] clean pruning code


## Quick Start

<details open>
<summary>Installation</summary>

Clone repo and install [requirements.txt](https://github.com/Laughing-q/yolov5-q/blob/master/requirements.txt) in a
**Python>=3.7.0** environment, including**PyTorch>=1.7.1**.

```shell
git clone https://github.com/Laughing-q/yolov5-q.git
cd yolov5-q
pip install -r requirements.txt
pip install -e .
```

</details>


<details open>
<summary>Training</summary>

- training objection
```shell
python tools/train.py --data ./data/seg/balloon.yaml --weights weights/yolov5s.pt --epochs 50 --batch-size 8
```

- training segmentation
```shell
python tools/train.py --data ./data/seg/balloon.yaml --weights weights/yolov5s.pt --cfg ./configs/segment/yolov5s_seg.yaml --epochs 50 --batch-size 8 --mask
```

</details>

<details open>
<summary>Evalution</summary>

- eval objection
```shell
python tools/val.py --data ./data/seg/balloon.yaml --weights weights/yolov5s.pt --epochs 50 --batch-size 8
```

- eval segmentation
```shell
python tools/val.py --data ./data/seg/balloon.yaml --weights weights/yolov5s.pt --batch-size 8 --mask
```

</details>


## Tips
- Plot mask will occupy a lot of cuda memory, so `plots=False` in `train_seg` by default, so you may need to run `val_seg.py` after running `train_seg.py` for more visualization.
- `process_mask` will save a lot of cuda memory, but get rough masks(`plots=False`).
- `process_mask_unsample` will occupy a lot of cuda memory, but get better masks(`plots=False`).
- not support `wandb` and `evolve`, cause I don't need them.
- Just put a `--mask` option and `--cfg` option, then you can train instance segmentation.
- Just put a `--mask` option, then you can val instance segmentation.

## Reference
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
