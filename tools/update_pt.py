from yolov5.models.yolo import Model
from yolov5.utils.torch_utils import select_device
import torch


def build_yolov5(cfg, weight_path, device, half=True):
    device = select_device(device)
    with torch.no_grad():
        model = Model(cfg).to(device)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt, strict=False)
        if half:
            model.half()
    return model


if __name__ == "__main__":
    model = build_yolov5(
        cfg="/home/laughing/code/yolov5/weights/yolov5n.yaml",
        weight_path="/home/laughing/code/yolov5/weights/yolov5n.pth",
        device="0",
        half=True,
    )
    ckpt = {"model": model}
    torch.save(ckpt, "yolov5n.pt")
