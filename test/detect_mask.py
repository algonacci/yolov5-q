from yolov5.core import Yolov5Segment
from tqdm import tqdm
import cv2


source = "/e/1.avi"

detector = Yolov5Segment(
    weights="/home/laughing/yolov5/runs/seg0301/coco_fix/weights/best.pt", device=0, img_hw=(384, 640)
)

cap = cv2.VideoCapture(source)
frames = 10000
pbar = tqdm(range(frames), total=frames)


def test_one_img():
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        outputs, masks = detector(frame)
        frame = detector.visualize(frame, outputs, masks)
        pbar.desc = f"{detector.times}"
        cv2.imshow("p", frame)
        if cv2.waitKey(1) == ord("q"):
            break


def test_multi_img():
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        outputs, masks = detector([frame, frame.copy()])
        frames = detector.visualize([frame, frame.copy()], outputs, masks)
        pbar.desc = f"{detector.times}"
        frame1, frame2 = frames
        cv2.imshow("p1", frame1)
        cv2.imshow("p2", frame2)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    test_one_img()
    # test_multi_img()
