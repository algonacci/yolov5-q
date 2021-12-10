from utils.torch_utils import select_device
import torch
import yaml
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight-path', type=str, default='/d/projects/workCode/guiyang_test/detectors/yolov5/weights/yolov5s.pt', help='pt weight path')
parser.add_argument('--save-dir', type=str, default='/d/projects/workCode/guiyang_test/detectors/yolov5/weights', help='dir to save pth and yaml file')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

opt = parser.parse_args()

weight_path = opt.weight_path
device = opt.device
save_dir = opt.save_dir

device = select_device(device)
weight_name = os.path.basename(weight_path)
save_path = osp.join(save_dir, weight_name)


with torch.no_grad():
    checkpoints = torch.load(weight_path, map_location=device)
    model = checkpoints['model'].float().eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.yaml['names'] = names
    with open(save_path.replace('.pt', '.yaml'), 'w') as f:
        yaml.safe_dump(model.yaml, f, sort_keys=False)
    torch.save(model.float().state_dict(), save_path.replace('.pt', '.pth'))
# model.float().fuse().eval()
