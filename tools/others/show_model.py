from yolov5.models.yolo import Model
import argparse
from pathlib import Path
import torch
from yolov5.utils.general import print_args
from yolov5.utils.checker import check_yaml
from yolov5.utils.torch_utils import select_device

FILE = Path(__file__).resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='yolov5n_seg.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', default=False, help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # print(model)

    # for k, m in model.named_modules():
    # # for k, m in model.named_children():
    #     print(k)
    #     # if isinstance(m, nn.BatchNorm2d):
    #         # print(m)

    model.train()
    # print(model.model[-1].proto_net)

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)
