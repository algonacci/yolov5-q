from yolov5.structures import Instances
import torch

instance = Instances()
instance.pred_bboxes = torch.randn((5, 4))
instance.pred_mask = torch.randn((5, 3))
print(instance.pred_bboxes)
print(instance.pred_mask)

print(instance[:, 1])
