# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path
from core.evaluator import Yolov5Evaluator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (
    check_yaml,
    check_requirements,
    set_logging,
    print_args,
)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    set_logging()
    check_requirements(exclude=("tensorboard", "thop"))
    evaluator = Yolov5Evaluator(
        data=opt.data,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.iou_thres,
        single_cls=opt.single_cls,
        augment=opt.augment,
        verbose=opt.verbose,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        half=opt.half,
    )

    if opt.task in ("train", "val", "test"):  # run normally
        evaluator.run(
            weights=opt.weights,
            batch_size=opt.batch_size,
            imgsz=opt.imgsz,
            save_txt=opt.save_txt,
            save_conf=opt.save_conf,
            save_json=opt.save_json,
            task=opt.task,
        )
    else:
        raise ValueError(f"not support task {opt.task}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
