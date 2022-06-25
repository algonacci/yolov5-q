from torch import nn
import torch
from ..builder import HEADS


# TODO: assign and loss
@HEADS.register()
class YOLOV5Head(nn.Module):
    onnx_dynamic = False  # ONNX export parameter

    def __init__(
        self,
        num_classes,
        anchors,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
    ):
        super().__init__()
        assert len(anchors) == len(strides) == len(in_channels)
        self.num_classes = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.strides = strides
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.num_layers  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        )  # shape(num_layers,num_anchors,2)

        self.preds = nn.ModuleList(
            nn.Conv2d(x, self.no * self.num_anchors, 1) for x in in_channels
        )  # output conv

    def forward(self, x):
        """
        Args:
            x (tuple(torch.Tensor)): tuple of FPN outputs.
        Returns:
            outputs or decode_outputs.
        """
        # print(self.anchors)
        outputs = []
        for conv, xi in zip(self.preds, x):
            xi = conv(xi)  # conv
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx = xi.shape
            xi = (
                xi.view(bs, self.num_anchors, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            outputs.append(xi)

        return outputs if self.training else self.decode_outputs(outputs)

    def decode_outputs(self, outputs):
        """Decode outputs"""
        decode_outputs = []  # inference output
        for i, output in enumerate(outputs):
            _, _, ny, nx, _ = output.shape
            if self.grid[i].shape[2:4] != output.shape[2:4] or self.onnx_dynamic:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = output.sigmoid()
            xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.strides[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy.type_as(y), wh.type_as(y), y[..., 4:]), -1)
            decode_outputs.append(y.view(-1, self.num_anchors * ny * nx, self.no))
        return torch.cat(decode_outputs, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid(
            [torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing="ij"
        )
        grid = torch.stack((xv, yv), 2).expand((1, self.num_anchors, ny, nx, 2)).float()
        anchor_grid = (
            # (self.anchors[i].clone() * self.strides[i])
            self.anchors[i]
            .clone()
            .view((1, self.num_anchors, 1, 1, 2))
            .expand((1, self.num_anchors, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid
