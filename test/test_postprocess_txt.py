"""
Read model's output from numpy(.npy) and do some postprocessing.
input:
    output1: (1, 3, 80, 80, 85)
    output2: (1, 3, 40, 40, 85)
    output3: (1, 3, 20, 20, 85)
"""

import numpy as np
import torch
import cv2
from yolov5.utils.boxes import non_max_suppression, scale_coords
from yolov5.utils.plots import Visualizer


class TestPost:
    def __init__(self, preds=[]) -> None:
        self.stride = [8, 16, 32]
        # self.preds = [torch.from_numpy(np.load(p)) for p in preds]
        self.preds = preds
        self.nl = len(self.preds)
        # for i in range(self.nl):
        #     b, h, w, _ = self.preds[i].shape
        #     self.preds[i] = self.preds[i].reshape((b, h, w, 3, 10)).permute(0, 3, 1, 2, 4)

        self.na = self.preds[0].shape[1]
        self.no = self.preds[0].shape[-1]

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.anchors = (
            torch.tensor(
                [
                    [10, 13, 16, 30, 33, 23],
                    [30, 61, 62, 45, 59, 119],
                    [116, 90, 156, 198, 373, 326],
                ]
            )
            .float()
            .view(self.nl, -1, 2)
        )

    def __call__(self):
        z = []
        for i in range(self.nl):
            _, _, ny, nx, _ = self.preds[i].shape  # (bs, 3, 20, 20, 85)
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            print(self.grid[i].shape, self.anchor_grid[i].shape)
            y = self.preds[i].sigmoid()
            xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy.type_as(y), wh.type_as(y), y[..., 4:]), -1)
            z.append(y.view(-1, self.na * ny * nx, self.no))
        output = torch.cat(z, 1)
        return output

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone())
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid


if __name__ == "__main__":
    # preds = [
    #     "./temp1/0.npy",
    #     "./temp1/1.npy",
    #     "./temp1/2.npy",
    # ]
    # preds = [
    #     "./temp2/0.npy",
    #     "./temp2/1.npy",
    #     "./temp2/2.npy",
    # ]

    output1 = np.loadtxt('/home/laughing/_1_3_52_96_32.txt')
    output1 = output1.reshape((1, 3, 52, 96, 32))
    output1 = torch.from_numpy(output1)

    output2 = np.loadtxt('/home/laughing/_1_3_26_48_32.txt')
    output2 = output2.reshape((1, 3, 26, 48, 32))
    output2 = torch.from_numpy(output2)

    output3 = np.loadtxt('/home/laughing/_1_3_13_24_32.txt')
    output3 = output3.reshape((1, 3, 13, 24, 32))
    output3 = torch.from_numpy(output3)

    # output1 = np.loadtxt('/home/laughing/1_3_20_20_85_.txt')
    # output1 = output1.reshape((1, 3, 20, 20, 85))
    # output1 = torch.from_numpy(output1)
    #
    # output2 = np.loadtxt('/home/laughing/1_3_40_40_85_.txt')
    # output2 = output2.reshape((1, 3, 40, 40, 85))
    # output2 = torch.from_numpy(output2)
    #
    # output3 = np.loadtxt('/home/laughing/1_3_80_80_85_.txt')
    # output3 = output3.reshape((1, 3, 80, 80, 85))
    # output3 = torch.from_numpy(output3)

    test = TestPost(preds=[output1, output2, output3])
    vis = Visualizer(names=list(range(80)))
    output = test()
    print(output.shape)
    output = non_max_suppression(output, conf_thres=0.2)
    print(output)

    img = cv2.imread('/home/laughing/code/yolov5-q/test/test_imgs/bus.jpg')

    for i, det in enumerate(output):  # detections per image
        if det is None or len(det) == 0:
            continue
        det[:, :4] = scale_coords((416, 768), det[:, :4], img.shape[:2]).round()

    print(output)
    img = vis(img, output, vis_confs=0.0)
    # cv2.imshow('p', cv2.resize(img, (1280, 704)))
    cv2.imshow('p', img)
    cv2.waitKey(0)
