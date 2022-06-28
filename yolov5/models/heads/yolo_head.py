from torch import nn
import torch
from ..builder import HEADS, build_loss
from yolov5.core import multi_apply
from yolov5.core.assigners import build_assigner


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
        train_cfg=None,
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=0.5
        ),
        loss_bbox=dict(
            type="IoULoss",
            reduction="sum",
            box_format="xywh",
            iou_type="ciou",
            loss_weight=0.05,
        ),
        loss_obj=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
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
        self.train_cfg = train_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)

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

    def loss(
        self,
        cls_preds,
        bbox_preds,
        objectnesses,
        gt_bboxes,
        gt_labels,
        img_metas,
    ):
        """label assign and Compute losses.

        Args:
            cls_preds (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W, num_classes).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W, 4).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W, 1).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [cx, cy, w, h] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        """
        lcls, lbox, lobj = (
            torch.zeros(1, device=cls_preds.device),
            torch.zeros(1, device=cls_preds.device),
            torch.zeros(1, device=cls_preds.device),
        )

        num_imgs = len(cls_preds)
        featmap_sizes = [cls_score.shape[2:4] for cls_score in cls_preds]
        flatten_cls_preds = [
            cls_pred.reshape(num_imgs, -1, self.num_classes) for cls_pred in cls_preds
        ]
        flatten_bbox_preds = [
            bbox_pred.reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.reshape(num_imgs, -1) for objectness in objectnesses
        ]

        # (batch, num_anchors*H*W, num_classes)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        # (batch, num_anchors*H*W, 4)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        # (batch, num_anchors*H*W, 1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)

        tcls, tbox, indices, anchors = multi_apply(
            self._get_target_single, gt_bboxes, gt_labels, [featmap_sizes] * num_imgs
        )

        indices = torch.cat(indices, 0)
        cls_targets = torch.cat(tcls, 0)
        bbox_targets = torch.cat(tbox, 0)
        anchors = torch.cat(anchors, 0)

        num_pos = len(bbox_targets)
        decoded_bboxes = self._decode_bbox(
            flatten_bbox_preds.view(-1, 4)[indices], anchors
        )
        # return iou for objectness
        loss_bbox, ious = self.loss_bbox(decoded_bboxes, bbox_targets, keep_iou=True)
        loss_bbox = loss_bbox / num_pos

        # TODO: gt_bboxes * img_size(e.g. 640)

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        feature_sizes,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [cx, cy, w, h] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            feature_sizes (list[(h, w), (h, w), (h, w)]): feature sizes
                of multi feature maps.
        """
        num_bboxes = gt_bboxes.shape[0]  # number of anchors, targets
        # (na, num_bboxes)
        anchor_idx = (
            torch.arange(self.num_anchors, device=gt_bboxes.device)
            .float()
            .view(self.num_anchors, 1)
            .repeat(1, num_bboxes)
        )  # same as .repeat_interleave(num_bboxes)
        bbox_idx = (
            torch.arange(num_bboxes, device=gt_bboxes.device)
            .float()
            .view(1, num_bboxes)
            .repeat(self.num_anchors, 1)
        )  # same as .repeat_interleave(num_bboxes)

        # targets.shape = (na, num_bboxes, 6), xywh + anchor_idx + bbox_index
        targets = torch.cat(
            (
                gt_bboxes.repeat(self.num_anchors, 1, 1),
                anchor_idx[:, :, None],
                bbox_idx[:, :, None],
            ),
            2,
        )  # append anchor indices

        tbox, tidx, indices, anch = self.assigner.assign(targets, feature_sizes)

        tcls = (
            torch.stack([gt_labels[t] for t in tidx], dim=0).view(-1)
            if len(tidx) > 0
            else tcls
        )
        anch = torch.stack(anch, dim=0).view(-1) if len(anch) > 0 else anch
        indices = torch.stack(indices, dim=0).view(-1) if len(indices) > 0 else indices
        tbox = torch.cat(tbox, dim=0).view(-1, 4) if len(tbox) > 0 else tbox

        return tcls, tbox, indices, anch

    def _decode_bbox(self, bbox_preds, anchors):
        """Decode the bbox preditions.
        Args:
            bbox_preds (Tensor): shape[num_assigned, 4].
            anchors (Tensor): anchors after label assign,
                which have same length of bbox_preds, shape[num_assigned, 2]
        """
        pxy = bbox_preds[:, :2].sigmoid() * 2.0 - 0.5
        pwh = (bbox_preds[:, 2:4].sigmoid() * 2) ** 2 * anchors
        pbox = torch.cat((pxy, pwh), 1)  # predicted box
        return pbox
