from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov5.core import Yolov5Evaluator
from yolov5.utils.plots import plot_one_box
import pycocotools.mask as mask_util
import json
import cv2
import os
from pathlib import Path


# rle = mask_util.encode(np.array(img[:, :, None], order="F", dtype="uint8"))[0]
# rle["counts"] = rle["counts"].decode("utf-8")


if __name__ == "__main__":

    evaluator = Yolov5Evaluator(
        data="./data/coco.yaml",
        conf_thres=0.001,
        iou_thres=0.6,
        exist_ok=False,
        half=True,
        mask=True,
    )

    evaluator.run(
        weights="./runs/seg0301/coco_s/weights/best.pt", batch_size=2, imgsz=640, save_json=True
        # weights="./weights/yolov5s.pt", batch_size=16, imgsz=640, save_json=True
    )

    # anno = COCO('/dataset/dataset/COCO/annotations/instances_val2017.json')  # init annotations api
    # pred = anno.loadRes('/home/laughing/code/yolov5-q/runs/val/exp14/predictions.json')  # init predictions api
    # eval = COCOeval(anno, pred, 'bbox')
    # # eval = COCOeval(anno, pred)
    # eval.params.imgIds = [int(Path(x).stem) for x in os.listdir('/home/laughing/code/yolov5-q/data/coco/images/val2017')]  # image IDs to evaluate
    # eval.evaluate()
    # eval.accumulate()
    # eval.summarize()

    # with open('/home/laughing/code/yolov5-q/runs/val/exp14/predictions.json', 'r') as f:
    #     preditions = json.load(f)
    # img_root = '/home/laughing/code/yolov5-q/data/coco/images/val2017'
    # print(preditions[0])
    # for p in preditions:
    #     box = p['bbox']
    #     x1, y1, w, h = box
    #     img_id = str(p['image_id']).zfill(12)
    #     img_name = f'{img_id}.jpg'
    #     img = cv2.imread(os.path.join(img_root, img_name))
    #     plot_one_box([x1, y1, x1 + w, y1 + h], img)
    #     cv2.imshow('p', img)
    #     print(box, p['score'])
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    #
