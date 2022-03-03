from ..utils.torch_utils import select_device
from ..models.experimental import attempt_load
from yolo.api.inference import Predictor
from yolo.api.visualization import Visualizer


class Yolov5:
    def __init__(self, weights, device, img_hw) -> None:
        model = attempt_load(weights)
        model = model.half()
        device = select_device(device)
        model.device = device
        model.model_type = 'yolov5'
        self.predictor = Predictor(img_hw=img_hw, models=model)
        self.vis = Visualizer(names=model.names)

    def __call__(self, images):
        """
        Args:
            images (List[np.ndarray] or np.ndarray): input images.
        """
        if not isinstance(images, list):
            images = [images]
        return self.predictor(images)

    def visualize(self, images, outputs, vis_confs=0.4):
        """Image visualize
        if images is a List of ndarray, then will return a List.
        if images is a ndarray , then return ndarray.
        """
        return self.vis(images, outputs, vis_confs)

if __name__ == '__main__':
    from tqdm import tqdm
    import cv2
    detector = Yolov5(weights='./weights/yolov5n.pt', device=0, img_hw=(384, 640))

    cap = cv2.VideoCapture("/e/1.avi")
    frames = 10000
    pbar = tqdm(range(frames), total=frames)

    for frame_num in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = detector(frame)
        detector.visualize(frame, outputs)
        cv2.imshow("p", frame)
        if cv2.waitKey(1) == ord("q"):
            break
