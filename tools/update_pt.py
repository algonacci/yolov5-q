from yolov5.models.yolo import Model
from yolov5.utils.torch_utils import select_device
import torch
import os.path as osp
import argparse

def build_yolov5(cfg, pth_path, device, half=True):
    device = select_device(device)
    with torch.no_grad():
        model = Model(cfg).to(device)
        ckpt = torch.load(pth_path)
        model.load_state_dict(ckpt, strict=False)
        if half:
            model.half()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='', help='yaml config path')
    parser.add_argument('-p', '--pth', type=str, default='', help='pth weight path')
    parser.add_argument('-s', '--save-dir', type=str, default='', help='dir to save new pt')
    parser.add_argument('-n', '--name', type=str, default='', help='dir to save new pt')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    pth = opt.pth
    cfg = opt.cfg or pth.replace('.pth', '.yaml')
    device = opt.device
    save_dir = opt.save_dir
    new_name = opt.name

    pth_name = osp.basename(pth)
    i = pth_name.rfind('.')
    weight_name = pth_name[:i]
    save_path = osp.join(save_dir, new_name or weight_name)

    model = build_yolov5(
        cfg=cfg,
        pth_path=pth,
        device=device,
        half=True,
    )
    ckpt = {"model": model}
    torch.save(ckpt, save_path + '.pt')
