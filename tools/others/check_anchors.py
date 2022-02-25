from yolov5.utils.autoanchor import check_anchors
from yolov5.data import LoadImagesAndLabels
from yolov5.utils.torch_utils import select_device
from yolov5.models.yolo import Model
from yolov5.utils.checker import check_yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="yolov5n_p4_tiny.yaml", help="model.yaml"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--data-path",
        default="/e/datasets/贵阳银行/play_phone/guiyang0721/images/train",
        help="data path",
    )
    parser.add_argument(
        "--imgsz",
        default=640,
        help="image size",
    )
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    device = select_device(opt.device)

    model = Model(opt.cfg).to(device)

    dataset = LoadImagesAndLabels(
        opt.data_path,
        opt.imgsz,
    )
    check_anchors(dataset, model=model, thr=4, imgsz=opt.imgsz)
