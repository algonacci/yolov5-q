# from yolov5.models.detectors import YOLOV5
from yolov5.models import build_detector
from yolov5.data.augmentations import letterbox
from yolov5.utils.boxes import non_max_suppression, scale_coords
from yolov5.utils.plots import Visualizer
from lqcv import Config
from collections import OrderedDict
import numpy as np
import torch
import cv2

anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
model_type = {
    "n": {"dep_mul": 0.33, "wid_mul": 0.25},
    "s": {"dep_mul": 0.33, "wid_mul": 0.5},
    "m": {"dep_mul": 0.67, "wid_mul": 0.75},
    "l": {"dep_mul": 1.0, "wid_mul": 1.0},
    "x": {"dep_mul": 1.33, "wid_mul": 1.25},
}
cfg = Config.fromfile('/home/laughing/codes/yolov5-q/configs/yolov5/yolov5n.py')
# new_model = YOLOV5(**model_type["n"], anchors=anchors)
new_model = build_detector(cfg.model)
new_model.eval()

model_dict = new_model.state_dict()
new_names = list(model_dict.keys())
new_dict = OrderedDict()


device = torch.device("cpu")
old_model = torch.load("weights/yolov5n.pt", map_location=device)
old_model = old_model["model"].float()  # load to FP32
old_model.eval()

old_dict = old_model.state_dict()
for i, (k, v) in enumerate(old_dict.items()):
    if "anchors" in k:
        new_dict[new_names[i]] = torch.tensor(anchors).float().view(3, -1, 2)
    else:
        new_dict[new_names[i]] = v

new_model.load_state_dict(new_dict)
new_model.eval()
vis = Visualizer(names=list(range(80)))
ori_img = cv2.imread("/home/laughing/codes/yolov5-q/data/images/bus.jpg")
h, w = ori_img.shape[:2]
img = letterbox(ori_img, (640, 480), auto=False)[0]
img = img.transpose(2, 0, 1)[::-1][None]
img = np.ascontiguousarray(img)
img = torch.from_numpy(img)
img = img / 255.0
output = new_model(img)
output = non_max_suppression(output)
for i, det in enumerate(output):  # detections per image
    if det is None or len(det) == 0:
        continue
    det[:, :4] = scale_coords((640, 480), det[:, :4], (h, w)).round()

vis(ori_img, output)
cv2.imshow("p", ori_img)
cv2.waitKey(0)

new_model.train()
outputs = new_model(img)
for output in new_model(img):
    print(output.shape)


# data_dict = {}
# key_list = []
# shape_list = []
# for k, v in model_dict.items():
#     vr = v.float().cpu().numpy()
#     key_list.append(k)
#     shape_list.append(vr.shape)
#
# with open('./weights/target.txt', 'w') as f:
#     for i, name in enumerate(key_list):
#         shape = ''
#         for item in shape_list[i]:
#             shape += str(item)
#             shape += ' '
#         f.write(name + ': ' + shape + '\n')
