# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (under development)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()

from yolov5.utils.general import (
    LOGGER, 
    colorstr,
    increment_path, 
    print_args, 
    strip_optimizer
)
from yolov5.utils.checker import (
    check_requirements,
)
from yolov5.utils.boxes import (
    xyxy2xywh,
    save_one_box
)
from yolov5.core import Yolov5Segment
from yolov5.data import create_reader


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        pause_det=False,
        ):
    pause = True
    source = str(source)
    save_img =not source.endswith('.txt')  # save inference images

    if nosave:
        save_img = False
        save_crop = False
        save_txt = False

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if not nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    detector = Yolov5Segment(weights, device=device, img_hw=imgsz, auto=True)
    names = detector.names

    # Dataloader
    dataset, webcam = create_reader(source)
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference

    vid_path, vid_writer = [None] * 1, [None] * 1

    # Run inference
    detector.warmup()  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    for image, path, s in dataset:
        image_copy = image.copy() if save_crop else image  # for save_crop

        # preprocess, inference, postprocess
        outputs, masks = detector.inference(image, conf_thres, iou_thres, classes, agnostic_nms)
        # time stuff
        dt[0] += detector.times['preprocess']
        dt[1] += detector.times['inference']
        dt[2] += detector.times['postprocess']

        # visualization
        if view_img or save_img:
            image = detector.visualize(image, outputs, masks)

        # Process predictions
        for i, det in enumerate(outputs):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, frame = path[i], dataset.count
                image_copy = image_copy[i]
                s += f'{i}: '
            else:
                p, frame = path, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % detector.img_hw  # print string

            if len(det):
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        save_one_box(xyxy, image_copy, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({detector.times["inference"]}ms)')

            # Stream results
            if view_img:
                if len(det) and pause_det:
                    pause = True
                cv2.namedWindow('p', cv2.WINDOW_NORMAL)
                # NOTE: just show one window if webcam.
                cv2.imshow('p', image[0] if webcam else image)
                key = cv2.waitKey(0 if pause else 1)
                pause = True if key == ord(' ') else False
                if key == ord('q') or key == ord('e') or key == 27:
                    exit()

            # Save results (image with detections)
            if save_img:
                save_path = str(save_dir / p.name)  # im.jpg
                if webcam:
                    # TODO: fix this fucking bug
                    # dataset.save(save_path, image[i], i)
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        fps, w, h = 30, image[i].shape[1], image[i].shape[0]
                        save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(image[i])
                else:
                    dataset.save(save_path, image)

    # Print results
    t = tuple(x / seen for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('-s', '--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-v', '--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('-p', '--pause-det', action='store_true', help='pasue the imshow when get some detections')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
