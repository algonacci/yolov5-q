import torch
from .builder import ASSIGNER


@ASSIGNER.register()
class YOLOV5Assginer(object):
    def __init__(
        self,
        anchor_thres=4,
        bias=0.5,
        offset=[
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ],
    ) -> None:
        self.anchor_t = anchor_thres
        self.bias = bias
        self.offset = offset

    def assign(
        self,
        targets,
        feature_sizes,
    ):
        """labels assign.
        Args:
            targets (Tensor): Ground truth bboxes, anchor index and
                label index of one image, a 2D-Tensor with
                shape [num_anchor, num_bboxes, 6],
                bboxes is [cx, cy, w, h] format.
            feature_sizes (list[(h, w), (h, w), (h, w)]): feature sizes
                of multi feature maps.
        Returns:
            tbox (list):  a list include gt bboxes after labels assign 
                from multi-level feature maps.
            tidx (list): a list include gt index after labels assign 
                from multi-level feature maps.
            indices (list): a list include preditions index after labels assign 
                from multi-level feature maps.
            anch (list):a list include anchors after labels assign 
                from multi-level feature maps.
        """
        tidx, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off = (
            torch.tensor(self.offset, device=targets.device).float() * self.bias
        )  # offsets

        for i in range(self.num_layers):
            anchors = self.anchors[i]
            gain[0:4] = torch.tensor(feature_sizes[i])[[1, 0, 1, 0]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            # Match targets to anchors
            if targets.shape[1]:  # num_bboxes
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1.0 < self.bias) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < self.bias) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = t[0]
                offsets = 0

            # Define
            gxy = t[:, 0:2]  # grid xy
            gwh = t[:, 2:4]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            aidx = t[:, 4].long()  # anchor indices
            indices.append(
                (
                    aidx * (gain[1] * gain[0])
                    + gj.clamp_(0, gain[1] - 1) * gain[1]
                    + gi.clamp_(0, gain[0] - 1)
                )
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[aidx])  # anchors
            tidx.append(t[:, 5].long())  # class
            # tcls.append(gt_labels[tidx])  # class

        return tbox, tidx, indices, anch
