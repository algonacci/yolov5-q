import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from numpy import random

from yolov5.models.experimental import attempt_load
from yolov5.data import letterbox
from yolov5.utils.boxes import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)

from yolov5.utils.checker import check_img_size

# ROIarea, polygon_ROIarea)
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device
import xml.dom.minidom

"""
将对图片的检测结果保存为xml，可以上传到华为云进行修改;
文件格式为source；
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    nargs="+",
    type=str,
    default="/home/laughing/yolov5/runs/jiujiang/1221/weights/best.pt",
    help="model.pt path(s)",
)
parser.add_argument(
    "--source", type=str, default="/d/九江/playphone/1218-1221_label", help="source"
)  # file/folder, 0 for webcam
parser.add_argument(
    "--output",
    type=str,
    default="/d/九江/playphone/1218-1221_label",
    help="output folder",
)  # output folder
parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
parser.add_argument(
    "--conf-thres", type=float, default=0.3, help="object confidence threshold"
)
parser.add_argument(
    "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
)
parser.add_argument(
    "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
)
parser.add_argument("--view-img", action="store_true", help="display results")
parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,  # default=[0, 1],
    help="filter by class: --class 0, or --class 0 2 3",
)
parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
parser.add_argument("--augment", action="store_true", help="augmented inference")
parser.add_argument("--update", action="store_true", help="update all models")
opt = parser.parse_args()

out, source, weights, imgsz = opt.output, opt.source, opt.weights, opt.img_size
webcam = (
    source.isnumeric()
    or source.startswith("rtsp")
    or source.startswith("http")
    or source.endswith(".txt")
)

device = select_device(opt.device)
half = True

imgs_list = os.listdir(source) if os.path.isdir(source) else [source]

poly_points = []
num_points = 4
num = 1
roi = False
# num = 100000000
pause = True


models, names, colors = [], [], []
weights = [weights] if not isinstance(weights, list) else weights
for weight in weights:
    # Load model
    model = attempt_load(weight, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    if half:
        model.half()  # to FP16

    # Get names and colors
    name = model.module.names if hasattr(model, "module") else model.names
    color = [
        [random.randint(0, 255) for _ in range(3)] for _ in range(len(name))
    ]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = (
        model(img.half() if half else img) if device.type != "cpu" else None
    )  # run once
    models.append(model)
    names.append(name)
    colors.append(color)


pbar = tqdm(imgs_list, total=len(imgs_list))
for img_name in pbar:
    pbar.desc = os.path.join(source, img_name)
    img0 = cv2.imread(os.path.join(source, img_name))
    # print(img0)
    # cv2.imshow('x', img0)
    # cv2.waitKey(0)
    if num == 1 and roi:
        for i in range(num_points):
            poly_points.append(
                cv2.selectROI(
                    windowName="roi", img=img0, showCrosshair=False, fromCenter=False
                )
            )
        cv2.destroyWindow("roi")
        poly_points.append(poly_points[0])
        poly_points = np.array(poly_points)

    img = letterbox(img0, new_shape=imgsz)[0]
    # cv2.imshow('x', img0)
    # cv2.waitKey(0)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    doc = xml.dom.minidom.Document()
    root = doc.createElement("annotation")

    node_folder = doc.createElement("folder")
    node_folder.appendChild(doc.createTextNode("jiujiang"))
    root.appendChild(node_folder)

    node_name = doc.createElement("filename")
    node_name.appendChild(doc.createTextNode(img_name))
    root.appendChild(node_name)

    node_source = doc.createElement("source")
    node_database = doc.createElement("database")
    node_database.appendChild(doc.createTextNode("Unknown"))
    node_source.appendChild(node_database)
    root.appendChild(node_source)

    node_path = doc.createElement("path")
    node_path.appendChild(doc.createTextNode("../JPEGImages/" + img_name))
    root.appendChild(node_path)

    node_size = doc.createElement("size")
    node_width = doc.createElement("width")
    node_width.appendChild(doc.createTextNode(str(img0.shape[1])))
    node_height = doc.createElement("height")
    node_height.appendChild(doc.createTextNode(str(img0.shape[0])))
    node_depth = doc.createElement("depth")
    node_depth.appendChild(doc.createTextNode(str(img0.shape[2])))
    node_size.appendChild(node_width)
    node_size.appendChild(node_height)
    node_size.appendChild(node_depth)
    root.appendChild(node_size)

    node_segmented = doc.createElement("segmented")
    node_segmented.appendChild(doc.createTextNode("0"))
    root.appendChild(node_segmented)

    im0 = img0.copy()
    for i, model in enumerate(models):
        pred = model(img, augment=opt.augment)[0]
        # print(pred)
        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        det = pred[0]

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            if len(poly_points) != 0:
                _, warn = polygon_ROIarea(det[:, :4], poly_points[:, :2], im0)
                det = det[warn]
            # print(det)

            for *xyxy, conf, cls in reversed(det):
                label = "%s %.2f" % (names[i][int(cls)], conf)
                plot_one_box(
                    xyxy, im0, label=label, color=colors[i][int(cls)], line_thickness=3
                )

                node_object = doc.createElement("object")
                node_name_object = doc.createElement("name")
                node_name_object.appendChild(doc.createTextNode(names[i][int(cls)]))
                # node_name_object.appendChild(doc.createTextNode('hair'))
                node_pose_object = doc.createElement("pose")
                node_pose_object.appendChild(doc.createTextNode("Unspecified"))

                node_truncated_object = doc.createElement("truncated")
                node_truncated_object.appendChild(doc.createTextNode("0"))

                node_difficult_object = doc.createElement("difficult")
                node_difficult_object.appendChild(doc.createTextNode("0"))

                node_occluded_object = doc.createElement("occluded")
                node_occluded_object.appendChild(doc.createTextNode("0"))

                node_bndbox_object = doc.createElement("bndbox")
                node_xmin_bndbox = doc.createElement("xmin")
                node_xmin_bndbox.appendChild(doc.createTextNode(str(int(xyxy[0]))))
                node_ymin_bndbox = doc.createElement("ymin")
                node_ymin_bndbox.appendChild(doc.createTextNode(str(int(xyxy[1]))))
                node_xmax_bndbox = doc.createElement("xmax")
                node_xmax_bndbox.appendChild(doc.createTextNode(str(int(xyxy[2]))))
                node_ymax_bndbox = doc.createElement("ymax")
                node_ymax_bndbox.appendChild(doc.createTextNode(str(int(xyxy[3]))))
                node_bndbox_object.appendChild(node_xmin_bndbox)
                node_bndbox_object.appendChild(node_ymin_bndbox)
                node_bndbox_object.appendChild(node_xmax_bndbox)
                node_bndbox_object.appendChild(node_ymax_bndbox)

                node_object.appendChild(node_name_object)
                node_object.appendChild(node_pose_object)
                node_object.appendChild(node_truncated_object)
                node_object.appendChild(node_difficult_object)
                node_object.appendChild(node_occluded_object)
                node_object.appendChild(node_bndbox_object)
                root.appendChild(node_object)

                with open(
                    os.path.join(out, os.path.splitext(img_name)[0] + ".xml"), "w"
                ) as fh:
                    # 将根节点添加到文档对象中
                    fh.truncate()
                    doc.appendChild(root)
                    doc.writexml(fh)
    num += 1
    if opt.view_img:
        cv2.imshow("xxx", im0)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(" ") else False
        if key == ord("q") or key == ord("e") or key == 27:
            raise StopIteration
