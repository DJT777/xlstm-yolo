# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ultralytics.models.utils.ops import get_cdn_group
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "ViLBasedRTDETRHead"


# #native dimensions input
# class Detect(nn.Module):
#     """YOLO Detect head for detection models."""

#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     format = None  # export format
#     end2end = False  # end2end
#     max_det = 300  # max_det
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#     legacy = False  # backward compatibility for v3/v5/v8/v9 models

#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLO detection layer with specified number of classes and channels."""
#         print(ch)
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build

#         # Define cv2 and cv3 to use input channels x directly as intermediate channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), nn.Conv2d(x, 4 * self.reg_max, 1)) for x in ch
#         )
#         self.cv3 = (
#             nn.ModuleList(
#                 nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), nn.Conv2d(x, self.nc, 1)) for x in ch
#             )
#             if self.legacy
#             else nn.ModuleList(
#                 nn.Sequential(
#                     nn.Sequential(DWConv(x, x, 3), Conv(x, x, 1)),
#                     nn.Sequential(DWConv(x, x, 3), Conv(x, x, 1)),
#                     nn.Conv2d(x, self.nc, 1),
#                 )
#                 for x in ch
#             )
#         )
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#         if self.end2end:
#             self.one2one_cv2 = copy.deepcopy(self.cv2)
#             self.one2one_cv3 = copy.deepcopy(self.cv3)

#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         if self.end2end:
#             return self.forward_end2end(x)

#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return x
#         y = self._inference(x)
#         return y if self.export else (y, x)

#     def forward_end2end(self, x):
#         """
#         Performs forward pass of the v10Detect module.

#         Args:
#             x (tensor): Input tensor.

#         Returns:
#             (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
#                            If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
#         """
#         x_detach = [xi.detach() for xi in x]
#         one2one = [
#             torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
#         ]
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return {"one2many": x, "one2one": one2one}

#         y = self._inference(one2one)
#         y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
#         return y if self.export else (y, {"one2many": x, "one2one": one2one})

#     def _inference(self, x):
#         """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
#         # Inference path
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.format != "imx" and (self.dynamic or self.shape != shape):
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape

#         if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
#             box = x_cat[:, : self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4 :]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

#         if self.export and self.format in {"tflite", "edgetpu"}:
#             # Precompute normalization factor to increase numerical stability
#             # See https://github.com/ultralytics/ultralytics/issues/7371
#             grid_h = shape[2]
#             grid_w = shape[3]
#             grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
#             norm = self.strides / (self.stride[0] * grid_size)
#             dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
#         elif self.export and self.format == "imx":
#             dbox = self.decode_bboxes(
#                 self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
#             )
#             return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
#         else:
#             dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

#         return torch.cat((dbox, cls.sigmoid()), 1)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
#         if self.end2end:
#             for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
#                 a[-1].bias.data[:] = 1.0  # box
#                 b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

#     def decode_bboxes(self, bboxes, anchors, xywh=True):
#         """Decode bounding boxes."""
#         return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

#     @staticmethod
#     def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
#         """
#         Post-processes YOLO model predictions.

#         Args:
#             preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
#                 format [x, y, w, h, class_probs].
#             max_det (int): Maximum detections per image.
#             nc (int, optional): Number of classes. Default: 80.

#         Returns:
#             (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
#                 dimension format [x, y, w, h, max_class_prob, class_index].
#         """
#         batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
#         boxes, scores = preds.split([4, nc], dim=-1)
#         index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
#         boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
#         scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
#         scores, index = scores.flatten(1).topk(min(max_det, anchors))
#         i = torch.arange(batch_size)[..., None]  # batch indices
#         return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


# original
class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        print(ch)
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]

        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
            
# class v10Detect(Detect):
#     """
#     Refactored v10 Detection head to use native input channel dimensions in intermediate layers.
#     This class can be dropped in place of the original v10Detect.
#     """
#     end2end = True

#     def __init__(self, nc=80, ch=()):
#         """
#         Initializes the Refactoredv10Detect object with the specified number of classes and input channels.

#         Args:
#             nc (int): Number of classes. Default is 80.
#             ch (tuple): Tuple of channel sizes for the input feature maps.
#         """
#         super().__init__(nc, ch)
#         # Light classification head using native input channel dimensions
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(
#                 nn.Sequential(Conv(x, x, 3, g=x), Conv(x, x, 1)),
#                 nn.Sequential(Conv(x, x, 3, g=x), Conv(x, x, 1)),
#                 nn.Conv2d(x, self.nc, 1),
#             )
#             for x in ch
#         )
#         self.one2one_cv3 = copy.deepcopy(self.cv3)

class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

from ultralytics.nn.modules.vision_lstm.vision_lstm2 import *
# head.py  ▸  ViLBasedRTDETRHead  (fully self-contained)

import warnings
import torch, torch.nn as nn
from torch.nn.init import constant_
from ultralytics.models.utils.ops import get_cdn_group
from .utils import bias_init_with_prob, linear_init

# class ViLBasedRTDETRHead(nn.Module):
#     export = False                     # keep Ultralytics export switch

#     # ------------------------------------------------------------------ #
#     #                           INITIALISATION                           #
#     # ------------------------------------------------------------------ #
#     def __init__(
#         self,
#         ch_vil,                        # e.g. (64,128,256)
#         vil_fpn_spatial_shapes,        # e.g. [[80,80],[40,40],[20,20]]
#         hidden_dim        = 256,
#         num_queries       = 300,
#         num_vil_self_attn_layers = 2,
#         vil_block_decoder_args = None,
#         act               = nn.SiLU,
#         nc                = 80,
#         num_denoising     = 0,
#         label_noise_ratio = 0.5,
#         box_noise_scale   = 1.0,
#         grid_size_anchor  = 0.05,
#         return_intermediate_outputs = False,
#     ):
#         super().__init__()

#         # ------------ static attributes ------------- #
#         self.nc, self.hidden_dim = int(nc), int(hidden_dim)
#         self.num_queries, self.num_denoising = int(num_queries), int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor

#         # ------------ input projection --------------- #
#         self.input_proj = nn.ModuleList(
#             [nn.Sequential(nn.Linear(c, hidden_dim), nn.LayerNorm(hidden_dim)) for c in ch_vil]
#         )
#         self.register_buffer(               # tiny tensor → moved automatically with the module
#             "vil_fpn_spatial_shapes", torch.tensor(vil_fpn_spatial_shapes, dtype=torch.long), persistent=False
#         )

#         # ------------ dense-proposal heads ----------- #
#         self.proposal_feature_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
#                                                         nn.LayerNorm(hidden_dim))
#         self.proposal_score_head = nn.Linear(hidden_dim, nc)
#         self.proposal_box_head   = MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU)

#         # ------------ ViL “decoder” ------------------- #
#         base_cfg = dict(
#             dim=hidden_dim, direction="ROWWISE_FROM_TOP_LEFT", conv_kind="causal1d",
#             conv_kernel_size=3, proj_bias=True, norm_bias=True, num_blocks=1,
#             init_weights="original", chunk_size=64,
#             qkv_block_size=max(1, hidden_dim // 64 if hidden_dim >= 64 else hidden_dim // 16),
#         )
#         base_cfg.update(vil_block_decoder_args or {})
#         if hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next((d for d in (16,32,64,128,256) if hidden_dim % d == 0), 1)
#         self.vil_self_attn_decoder = nn.ModuleList(
#             [ViLBlock(**base_cfg) for _ in range(num_vil_self_attn_layers)]
#         )

#         # ------------ prediction heads --------------- #
#         n_sets = 1 if not return_intermediate_outputs else max(1, num_vil_self_attn_layers)
#         self.final_score_heads = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(n_sets))
#         self.final_box_heads   = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets))

#         # ------------ optional denoising ------------- #
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale

#         # ------------ query positional enc ----------- #
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=nn.SiLU)

#         self._reset_parameters()

#     # ------------------------------------------------------------------ #
#     #                        PARAMETER INIT                               #
#     # ------------------------------------------------------------------ #
#     def _reset_parameters(self):
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
#         for mod in self.modules():
#             if isinstance(mod, nn.Linear) and mod.weight.requires_grad:
#                 nn.init.xavier_uniform_(mod.weight)
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#     # ------------------------------------------------------------------ #
#     # helpers: anchors & meshgrid                                         #
#     # ------------------------------------------------------------------ #
#     @staticmethod
#     def _meshgrid(h, w, dtype, device):
#         y, x = torch.arange(h, dtype=dtype, device=device), torch.arange(w, dtype=dtype, device=device)
#         gy, gx = torch.meshgrid(y, x, indexing="ij")
#         return gx, gy

#     def _generate_anchors_cxcywh_norm(self, shapes, dtype, device):
#         out = []
#         for i, (h, w) in enumerate(shapes):
#             gx, gy = self._meshgrid(int(h), int(w), dtype, device)
#             cxcy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
#             wh   = torch.full_like(cxcy, self.grid_size_anchor * (2.0 ** i))
#             out.append(torch.cat([cxcy, wh], -1).view(-1, 4))
#         return torch.cat(out).unsqueeze(0)          # (1, ΣHW, 4)

#     # ------------------------------------------------------------------ #
#     # stage-1: dense proposals ( + optional CDN )                         #
#     # ------------------------------------------------------------------ #
#     def _get_initial_proposals(self, feats, batch):
#         seqs = [
#             proj(f.permute(0,2,3,1).reshape(f.size(0), -1, f.size(1))) for f, proj in zip(feats, self.input_proj)
#         ]
#         cat  = torch.cat(seqs, 1)                                   # (B, Ntok, D)
#         bs, ntok, _ = cat.shape

#         anchors = self._generate_anchors_cxcywh_norm(
#             self.vil_fpn_spatial_shapes, cat.dtype, cat.device
#         ).repeat(bs,1,1)
#         dense_feat = self.proposal_feature_transform(cat)
#         logits     = self.proposal_score_head(dense_feat)           # (B, Ntok, C)
#         deltas     = self.proposal_box_head  (dense_feat)           # (B, Ntok, 4)

#         ref_cxcy, ref_wh = anchors[...,:2], anchors[...,2:]
#         boxes = torch.cat([
#             ref_cxcy + deltas[...,:2] * ref_wh,
#             ref_wh  * torch.exp(deltas[...,2:])
#         ], -1).sigmoid()

#         topk = logits.sigmoid().amax(-1).topk(min(self.num_queries, ntok), 1).indices
#         gather = lambda t,d: torch.gather(t,1,topk.unsqueeze(-1).expand(-1,-1,d))
#         sel_feat  = gather(cat  , self.hidden_dim)
#         sel_box   = gather(boxes, 4)
#         sel_score = gather(logits, self.nc)

#         dn_meta = dict(
#             num_denoising_queries = 0,
#             num_matching_queries  = sel_feat.size(1),
#             dn_num_split          = [0, sel_feat.size(1)],
#             dn_pos_idx            = [],
#             dn_num_group          = 0,
#         )

#         # optional CDN (denoising) – only in training
#         dn_emb = dn_bbox = attn_mask = None
#         if self.training and self.num_denoising:
#             try:
#                 dn_emb, dn_bbox, attn_mask, extra = get_cdn_group(
#                     batch, self.nc, sel_feat.size(1),
#                     self.denoising_class_embed.weight,
#                     self.num_denoising, self.label_noise_ratio,
#                     self.box_noise_scale, True
#                 )
#                 dn_meta.update(extra)
#                 dn_meta["num_denoising_queries"] = dn_emb.size(1)
#                 dn_meta["dn_num_split"] = [dn_emb.size(1), sel_feat.size(1)]
#             except Exception as e:
#                 warnings.warn(f"CDN disabled: {e}")

#         qry_feat = torch.cat([dn_emb, sel_feat],1) if dn_emb is not None else sel_feat
#         qry_box  = torch.cat([dn_bbox, sel_box],1) if dn_bbox is not None else sel_box

#         return qry_feat, qry_box, sel_score, sel_box, attn_mask, dn_meta

#     # ------------------------------------------------------------------ #
#     #                           FORWARD                                   #
#     # ------------------------------------------------------------------ #
#     def forward(self, x, batch=None):
#         qf, qb, enc_scores, enc_boxes, _, dn_meta = \
#             self._get_initial_proposals(x, batch if self.training else None)

#         # ------ make sure everything is same dtype (AMP-safe) ------------ #
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         qf, qb = qf.to(tgt_dtype), qb.to(tgt_dtype)

#         # ------ ViL decoder --------------------------------------------- #
#         q = qf + self.query_pos_head(qb.detach())
#         cls_out, box_out = [], []
#         for i, layer in enumerate(self.vil_self_attn_decoder):
#             q = layer(q)
#             idx = i if i < len(self.final_box_heads) else 0
#             cls_out.append(self.final_score_heads[idx](q))
#             box_out.append(self.final_box_heads[idx](q).sigmoid())
#         dec_scores, dec_boxes = torch.stack(cls_out), torch.stack(box_out)

#         # ensure split info present
#         dn_meta.setdefault("dn_num_split", [
#             dn_meta.get("num_denoising_queries", 0),
#             dn_meta.get("num_matching_queries",  0)
#         ])

#         # ---------------------- training path --------------------------- #
#         if self.training:
#             k = dn_meta["num_matching_queries"]
#             return dec_boxes, dec_scores, enc_boxes[:, :k], enc_scores[:, :k], dn_meta

#         # ---------------------- inference / validation ------------------ #
#         # ‹dn_meta=None› prevents the RT-DETR loss from trying to split dn queries
#         dn_meta = None
#         final_boxes, final_scores = dec_boxes[-1], dec_scores[-1].sigmoid()
#         out = torch.cat([final_boxes, final_scores], -1)             # (B, N, 4+C)
#         return out if self.export else (out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta))



from .vision_lstm.vision_lstm2 import *
# class ViLBasedRTDETRHead(nn.Module):
#     """RT-DETR head using ViL ‘writers’ + xLSTM ‘readers’ (linear-time cross-attention)."""
#     export = False

#     def __init__(
#         self,
#         ch_vil,                     # (64,128,256)
#         vil_fpn_spatial_shapes,     # [[80,80],[40,40],[20,20]]
#         hidden_dim        = 256,
#         num_queries       = 300,
#         num_vil_self_attn_layers = 2,
#         vil_block_decoder_args = None,
#         act               = nn.SiLU, 
#         nc                = 80,
#         num_denoising     = 0,
#         label_noise_ratio = 0.5,
#         box_noise_scale   = 1.0,
#         grid_size_anchor  = 0.05,
#         return_intermediate_outputs = False,
#     ):
#         super().__init__()
#         print(num_queries)
#         print(num_denoising)

#         # ── constants ───────────────────────────────────────────────────
#         self.nc, self.hidden_dim = int(nc), int(hidden_dim)
#         self.num_queries, self.num_denoising = int(num_queries), int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor

#         # ── 1. ViL-FPN projection (C,H,W) → (HW,D)  ---------------------
#         self.input_proj = nn.ModuleList(
#             nn.Sequential(nn.Linear(c, hidden_dim), nn.LayerNorm(hidden_dim)) for c in ch_vil
#         )
#         self.register_buffer("vil_fpn_spatial_shapes",
#                              torch.tensor(vil_fpn_spatial_shapes, dtype=torch.long),
#                              persistent=False)

#         # ── 2. dense-proposal stage (unchanged vs RT-DETR) --------------
#         self.proposal_feature_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
#                                                         nn.LayerNorm(hidden_dim))
#         self.proposal_score_head = nn.Linear(hidden_dim, nc)
#         self.proposal_box_head   = MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU)

#         # ── 3. ViL “writers” -------------------------------------------
#         base_cfg = dict(
#             dim=hidden_dim, direction="ROWWISE_FROM_TOP_LEFT",
#             conv_kind="causal1d", conv_kernel_size=3,
#             proj_bias=True, norm_bias=True, num_blocks=1,
#             init_weights="original", chunk_size=64,
#             qkv_block_size=max(1, hidden_dim // 64 if hidden_dim >= 64 else hidden_dim // 16),
#         )
#         base_cfg.update(vil_block_decoder_args or {})
#         if hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next((d for d in (16,32,64,128,256) if hidden_dim % d == 0), 1)

#         self.writer = nn.ModuleList(ViLBlock(**base_cfg) for _ in range(num_vil_self_attn_layers))

#         # ── 4. xLSTM cross “readers” ------------------------------------
#         self.reader = nn.ModuleList(
#             xLSTMCrossReader(
#                 dim              = hidden_dim,
#                 expansion        = base_cfg["num_blocks"],
#                 qkv_block_size   = base_cfg["qkv_block_size"],
#                 conv_kind        = base_cfg["conv_kind"],
#                 conv_kernel_size = base_cfg["conv_kernel_size"],
#                 chunk_size       = base_cfg["chunk_size"],
#                 gate_soft_cap    = base_cfg.get("gate_soft_cap", 15.0),
#             ) for _ in range(num_vil_self_attn_layers)
#         )

#         # ── 5. prediction heads (aux-loss aware) ------------------------
#         n_sets = max(1, num_vil_self_attn_layers) if return_intermediate_outputs else 1
#         self.final_score_heads = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(n_sets))
#         self.final_box_heads   = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets))

#         # ── 6. denoising -----------------------------------------------
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale

#         # ── 7. query positional encoding -------------------------------
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=nn.SiLU)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.weight.requires_grad:
#                 nn.init.xavier_uniform_(m.weight)
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#     @staticmethod
#     def _meshgrid(H, W, dtype, device):
#         y, x = torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device)
#         gy, gx = torch.meshgrid(y, x, indexing="ij")
#         return gx, gy

#     def _anchors(self, dtype, device):
#         shapes = self.vil_fpn_spatial_shapes.to(device)           # ensure same device
#         out = []
#         for i, (H, W) in enumerate(shapes):
#             gx, gy = self._meshgrid(int(H), int(W), dtype, device)
#             cxcy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
#             wh   = torch.full_like(cxcy, self.grid_size_anchor * (2.0 ** i))
#             out.append(torch.cat([cxcy, wh], -1).view(-1, 4))
#         return torch.cat(out).unsqueeze(0)  # (1, ΣHW, 4)

#     def _initial_queries(self, feats, batch):
#         seqs = [
#             proj_layer(f.permute(0, 2, 3, 1).reshape(f.size(0), -1, f.size(1)))
#             for f, proj_layer in zip(feats, self.input_proj)
#         ]
#         cat = torch.cat(seqs, 1)
#         bs, ntok, _ = cat.shape
#         anchors = self._anchors(cat.dtype, cat.device).repeat(bs, 1, 1)

#         dense = self.proposal_feature_transform(cat)
#         score = self.proposal_score_head(dense)
#         delta = self.proposal_box_head(dense)

#         ref_cxcy, ref_wh = anchors[..., :2], anchors[..., 2:]
#         boxes = torch.cat([ref_cxcy + delta[..., :2] * ref_wh,
#                         ref_wh * torch.exp(delta[..., 2:])], -1).sigmoid()

#         k = max(1, min(self.num_queries, ntok))
#         topk = score.sigmoid().amax(-1).topk(k, 1).indices
#         gather = lambda t, d: torch.gather(t, 1, topk.unsqueeze(-1).expand(-1, -1, d))
#         sel_feat, sel_box, sel_score = gather(cat, self.hidden_dim), gather(boxes, 4), gather(score, self.nc)

#         # Properly initialized default dn_meta (ensuring compatibility with RT-DETR loss function)
#         dn_meta = dict(
#             num_denoising_queries=0,
#             num_matching_queries=k,
#             dn_num_split=[0, k],
#             dn_pos_idx=[torch.zeros([0], dtype=torch.long, device=cat.device) for _ in range(bs)],
#             dn_num_group=0
#         )

#         dn_emb = dn_bbox = None
#         if self.training and self.num_denoising:
#             try:
#                 dn_emb, dn_bbox, _, extra = get_cdn_group(
#                     batch, self.nc, k, self.denoising_class_embed.weight,
#                     self.num_denoising, self.label_noise_ratio, self.box_noise_scale, True
#                 )
#                 dn_meta.update(extra)
#                 dn_meta["num_denoising_queries"] = dn_emb.size(1)
#                 dn_meta["dn_num_split"] = [dn_emb.size(1), k]
#             except Exception as e:
#                 warnings.warn(f"CDN disabled: {e}")

#         q_feat = torch.cat([dn_emb, sel_feat], 1) if dn_emb is not None else sel_feat
#         q_box = torch.cat([dn_bbox, sel_box], 1) if dn_bbox is not None else sel_box
#         return q_feat, q_box, sel_score, sel_box, dn_meta

#     def forward(self, x, batch=None):
#         qf, qb, enc_scores, enc_boxes, dn_meta = (*self._initial_queries(x, batch if self.training else None),)
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         q = qf.to(tgt_dtype) + self.query_pos_head(qb.to(tgt_dtype).detach())
#         cls_all, box_all = [], []
#         for i in range(len(self.writer)):
#             memory = self.writer[i](q)
#             q = self.reader[i](memory, q)
#             head_idx = i if i < len(self.final_box_heads) else 0
#             cls_all.append(self.final_score_heads[head_idx](q))
#             box_all.append(self.final_box_heads[head_idx](q).sigmoid())
#         dec_scores, dec_boxes = torch.stack(cls_all), torch.stack(box_all)

#         k = dn_meta["num_matching_queries"]

#         if self.training:
#             return dec_boxes, dec_scores, enc_boxes[:, :k], enc_scores[:, :k], dn_meta
#         else:
#             if dn_meta["num_denoising_queries"] > 0:
#                 offset = dn_meta["num_denoising_queries"]
#                 out = torch.cat([dec_boxes[-1][:, offset:], dec_scores[-1][:, offset:].sigmoid()], -1)
#             else:
#                 out = torch.cat([dec_boxes[-1], dec_scores[-1].sigmoid()], -1)
#             dn_meta = None  # Set dn_meta to None during validation
#             return out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)



# class ViLBasedRTDETRHead(nn.Module):
#     """RT-DETR head using ViLLayerFusedQKV writers + strict-masking xLSTM readers with directionality."""
#     export = False

#     def __init__(
#         self,
#         ch_vil,
#         vil_fpn_spatial_shapes,
#         hidden_dim=256,
#         num_queries=300,
#         num_vil_self_attn_layers=2,
#         vil_block_decoder_args=None,
#         act=nn.SiLU, 
#         nc=80,
#         num_denoising=0,
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         grid_size_anchor=0.05,
#         return_intermediate_outputs=False,
#     ):
#         super().__init__()
#         self.nc, self.hidden_dim = int(nc), int(hidden_dim)
#         self.num_queries, self.num_denoising = int(num_queries), int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor

#         self.input_proj = nn.ModuleList(
#             nn.Sequential(nn.Linear(c, hidden_dim), nn.LayerNorm(hidden_dim)) for c in ch_vil
#         )
#         self.register_buffer("vil_fpn_spatial_shapes",
#                              torch.tensor(vil_fpn_spatial_shapes, dtype=torch.long),
#                              persistent=False)

#         self.proposal_feature_transform = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
#         )
#         self.proposal_score_head = nn.Linear(hidden_dim, nc)
#         self.proposal_box_head   = MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU)

#         base_cfg = dict(
#             dim=hidden_dim,
#             expansion=2,
#             qkv_block_size=max(1, hidden_dim // 64 if hidden_dim >= 64 else hidden_dim // 16),
#             proj_bias=True,
#             norm_bias=True,
#             num_blocks=1,
#             gate_soft_cap=15.0,
#             chunk_size=64,
#         )
#         if vil_block_decoder_args is not None:
#             base_cfg.update(vil_block_decoder_args)
#         if hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next(
#                 (d for d in (16, 32, 64, 128, 256) if hidden_dim % d == 0), 1
#             )

#         # Alternate direction for each layer (even=forward, odd=reverse)
#         self.writer = nn.ModuleList(
#             ViLLayerFusedQKV(
#                 **base_cfg,
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(num_vil_self_attn_layers)
#         )
#         self.reader = nn.ModuleList(
#             RefactoredxLSTMCrossReader(
#                 dim=hidden_dim,
#                 expansion=base_cfg["expansion"],
#                 qkv_block_size=base_cfg["qkv_block_size"],
#                 chunk_size=base_cfg["chunk_size"],
#                 gate_soft_cap=base_cfg["gate_soft_cap"],
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(num_vil_self_attn_layers)
#         )

#         n_sets = max(1, num_vil_self_attn_layers) if return_intermediate_outputs else 1
#         self.final_score_heads = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(n_sets))
#         self.final_box_heads = nn.ModuleList(
#             MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets)
#         )
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=nn.SiLU)
#         self._reset_parameters()


#     def _reset_parameters(self):
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.weight.requires_grad:
#                 nn.init.xavier_uniform_(m.weight)
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#     @staticmethod
#     def _meshgrid(H, W, dtype, device):
#         y, x = torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device)
#         gy, gx = torch.meshgrid(y, x, indexing="ij")
#         return gx, gy

#     def _anchors(self, dtype, device):
#         shapes = self.vil_fpn_spatial_shapes.to(device)
#         out = []
#         for i, (H, W) in enumerate(shapes):
#             gx, gy = self._meshgrid(int(H), int(W), dtype, device)
#             cxcy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
#             wh   = torch.full_like(cxcy, self.grid_size_anchor * (2.0 ** i))
#             out.append(torch.cat([cxcy, wh], -1).view(-1, 4))
#         return torch.cat(out).unsqueeze(0)

#     def _initial_queries(self, feats, batch):
#         seqs = [
#             proj_layer(f.permute(0, 2, 3, 1).reshape(f.size(0), -1, f.size(1)))
#             for f, proj_layer in zip(feats, self.input_proj)
#         ]
#         cat = torch.cat(seqs, 1)
#         bs, ntok, _ = cat.shape
#         anchors = self._anchors(cat.dtype, cat.device).repeat(bs, 1, 1)

#         dense = self.proposal_feature_transform(cat)
#         score = self.proposal_score_head(dense)
#         delta = self.proposal_box_head(dense)

#         ref_cxcy, ref_wh = anchors[..., :2], anchors[..., 2:]
#         boxes = torch.cat([ref_cxcy + delta[..., :2] * ref_wh,
#                            ref_wh * torch.exp(delta[..., 2:])], -1).sigmoid()

#         k = max(1, min(self.num_queries, ntok))
#         topk = score.sigmoid().amax(-1).topk(k, 1).indices
#         gather = lambda t, d: torch.gather(t, 1, topk.unsqueeze(-1).expand(-1, -1, d))
#         sel_feat, sel_box, sel_score = gather(cat, self.hidden_dim), gather(boxes, 4), gather(score, self.nc)

#         dn_meta = dict(
#             num_denoising_queries=0,
#             num_matching_queries=k,
#             dn_num_split=[0, k],
#             dn_pos_idx=[torch.zeros([0], dtype=torch.long, device=cat.device) for _ in range(bs)],
#             dn_num_group=0
#         )

#         dn_emb = dn_bbox = None
#         if self.training and self.num_denoising:
#             try:
#                 dn_emb, dn_bbox, _, extra = get_cdn_group(
#                     batch, self.nc, k, self.denoising_class_embed.weight,
#                     self.num_denoising, self.label_noise_ratio, self.box_noise_scale, True
#                 )
#                 dn_meta.update(extra)
#                 dn_meta["num_denoising_queries"] = dn_emb.size(1)
#                 dn_meta["dn_num_split"] = [dn_emb.size(1), k]
#             except Exception as e:
#                 warnings.warn(f"CDN disabled: {e}")

#         q_feat = torch.cat([dn_emb, sel_feat], 1) if dn_emb is not None else sel_feat
#         q_box = torch.cat([dn_bbox, sel_box], 1) if dn_bbox is not None else sel_box

#         if self.training:
#             q_feat = q_feat.detach()
#             q_box = q_box.detach()

#         return q_feat, q_box, sel_score, sel_box, dn_meta

#     def forward(self, x, batch=None):
#         # Prepare queries as before
#         qf, qb, enc_scores, enc_boxes, dn_meta = (*self._initial_queries(x, batch if self.training else None),)
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         q = qf.to(tgt_dtype) + self.query_pos_head(qb.to(tgt_dtype))
#         cls_all, box_all = [], []

#         # Prepare memory ONCE for all layers (project all FPN feature maps)
#         projected_feats = [
#             proj_layer(f.permute(0, 2, 3, 1).reshape(f.size(0), -1, f.size(1)))
#             for f, proj_layer in zip(x, self.input_proj)
#         ]
#         memory = torch.cat(projected_feats, 1)  # (B, N_mem, hidden_dim)

#         for i in range(len(self.writer)):
#             q = self.writer[i](q)                        # self-attn on queries (ViLLayer)
#             q = self.reader[i](memory=memory, queries=q) # cross-attn to image memory (xLSTM)
#             head_idx = i if i < len(self.final_box_heads) else 0
#             cls_all.append(self.final_score_heads[head_idx](q))
#             box_all.append(self.final_box_heads[head_idx](q).sigmoid())
#         dec_scores, dec_boxes = torch.stack(cls_all), torch.stack(box_all)

#         k = dn_meta["num_matching_queries"]
#         if self.training:
#             return dec_boxes, dec_scores, enc_boxes[:, :k], enc_scores[:, :k], dn_meta
#         else:
#             if dn_meta["num_denoising_queries"] > 0:
#                 offset = dn_meta["num_denoising_queries"]
#                 out = torch.cat([dec_boxes[-1][:, offset:], dec_scores[-1][:, offset:].sigmoid()], -1)
#             else:
#                 out = torch.cat([dec_boxes[-1], dec_scores[-1].sigmoid()], -1)
#             dn_meta = None
#             return out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)


# class ViLBasedRTDETRHead(nn.Module):
#     """RT-DETR head using ViLLayerFusedQKV writers + strict-masking xLSTM readers with directionality."""
#     export = False

#     def __init__(
#         self,
#         ch_vil,
#         vil_fpn_spatial_shapes,
#         hidden_dim=256,
#         num_queries=300,
#         num_vil_self_attn_layers=2,
#         vil_block_decoder_args=None,
#         act=nn.SiLU, 
#         nc=80,
#         num_denoising=0,
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         grid_size_anchor=0.05,
#         return_intermediate_outputs=False,
#     ):
#         super().__init__()
#         self.nc, self.hidden_dim = int(nc), int(hidden_dim)
#         self.num_queries, self.num_denoising = int(num_queries), int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor

#         self.input_proj = nn.ModuleList(
#             nn.Sequential(nn.Linear(c, hidden_dim), nn.LayerNorm(hidden_dim)) for c in ch_vil
#         )
#         self.register_buffer("vil_fpn_spatial_shapes",
#                              torch.tensor(vil_fpn_spatial_shapes, dtype=torch.long),
#                              persistent=False)

#         self.proposal_feature_transform = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
#         )
#         self.proposal_score_head = nn.Linear(hidden_dim, nc)
#         self.proposal_box_head   = MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU)

#         base_cfg = dict(
#             dim=hidden_dim,
#             expansion=2,
#             qkv_block_size=max(1, hidden_dim // 64 if hidden_dim >= 64 else hidden_dim // 16),
#             proj_bias=True,
#             norm_bias=True,
#             num_blocks=1,
#             gate_soft_cap=15.0,
#             chunk_size=64,
#         )
#         if vil_block_decoder_args is not None:
#             base_cfg.update(vil_block_decoder_args)
#         if hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next(
#                 (d for d in (16, 32, 64, 128, 256) if hidden_dim % d == 0), 1
#             )

#         self.writer = nn.ModuleList(
#             ViLLayerFusedQKV(
#                 **base_cfg,
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(num_vil_self_attn_layers)
#         )
#         self.reader = nn.ModuleList(
#             RefactoredxLSTMCrossReader(
#                 dim=hidden_dim,
#                 expansion=base_cfg["expansion"],
#                 qkv_block_size=base_cfg["qkv_block_size"],
#                 chunk_size=base_cfg["chunk_size"],
#                 gate_soft_cap=base_cfg["gate_soft_cap"],
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(num_vil_self_attn_layers)
#         )

#         n_sets = max(1, num_vil_self_attn_layers) if return_intermediate_outputs else 1
#         self.final_score_heads = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(n_sets))
#         self.final_box_heads = nn.ModuleList(
#             MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets)
#         )
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=nn.SiLU)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.weight.requires_grad:
#                 nn.init.xavier_uniform_(m.weight)
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#     @staticmethod
#     def _meshgrid(H, W, dtype, device):
#         y, x = torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device)
#         gy, gx = torch.meshgrid(y, x, indexing="ij")
#         return gx, gy

#     def _anchors(self, dtype, device):
#         shapes = self.vil_fpn_spatial_shapes.to(device)
#         out = []
#         for i, (H, W) in enumerate(shapes):
#             gx, gy = self._meshgrid(int(H), int(W), dtype, device)
#             cxcy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
#             wh   = torch.full_like(cxcy, self.grid_size_anchor * (2.0 ** i))
#             out.append(torch.cat([cxcy, wh], -1).view(-1, 4))
#         return torch.cat(out).unsqueeze(0)

#     def _initial_queries(self, feats, batch):
#         seqs = [
#             proj_layer(f.permute(0, 2, 3, 1).reshape(f.size(0), -1, f.size(1)))
#             for f, proj_layer in zip(feats, self.input_proj)
#         ]
#         cat = torch.cat(seqs, 1)
#         bs, ntok, _ = cat.shape
#         anchors = self._anchors(cat.dtype, cat.device).repeat(bs, 1, 1)

#         dense = self.proposal_feature_transform(cat)
#         score = self.proposal_score_head(dense)
#         delta = self.proposal_box_head(dense)

#         ref_cxcy, ref_wh = anchors[..., :2], anchors[..., 2:]
#         boxes = torch.cat([ref_cxcy + delta[..., :2] * ref_wh,
#                            ref_wh * torch.exp(delta[..., 2:])], -1).sigmoid()

#         k = max(1, min(self.num_queries, ntok))
#         topk = score.sigmoid().amax(-1).topk(k, 1).indices
#         gather = lambda t, d: torch.gather(t, 1, topk.unsqueeze(-1).expand(-1, -1, d))
#         sel_feat, sel_box, sel_score = gather(cat, self.hidden_dim), gather(boxes, 4), gather(score, self.nc)

#         dn_meta = dict(
#             num_denoising_queries=0,
#             num_matching_queries=k,
#             dn_num_split=[0, k],
#             dn_pos_idx=[torch.zeros([0], dtype=torch.long, device=cat.device) for _ in range(bs)],
#             dn_num_group=0
#         )

#         dn_emb = dn_bbox = None
#         if self.training and self.num_denoising:
#             try:
#                 dn_emb, dn_bbox, _, extra = get_cdn_group(
#                     batch, self.nc, k, self.denoising_class_embed.weight,
#                     self.num_denoising, self.label_noise_ratio, self.box_noise_scale, True
#                 )
#                 dn_meta.update(extra)
#                 dn_meta["num_denoising_queries"] = dn_emb.size(1)
#                 dn_meta["dn_num_split"] = [dn_emb.size(1), k]
#             except Exception as e:
#                 warnings.warn(f"CDN disabled: {e}")

#         q_feat = torch.cat([dn_emb, sel_feat], 1) if dn_emb is not None else sel_feat
#         q_box = torch.cat([dn_bbox, sel_box], 1) if dn_bbox is not None else sel_box

#         if self.training:
#             q_feat = q_feat.detach()
#             q_box = q_box.detach()

#         return q_feat, q_box, sel_score, sel_box, dn_meta

#     def forward(self, x, batch=None):
#         # Prepare queries as before
#         qf, qb, enc_scores, enc_boxes, dn_meta = (*self._initial_queries(x, batch if self.training else None),)
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         q = qf.to(tgt_dtype) + self.query_pos_head(qb.to(tgt_dtype))

#         # Prepare memory ONCE for all layers (project all FPN feature maps)
#         projected_feats = [
#             proj_layer(f.permute(0, 2, 3, 1).reshape(f.size(0), -1, f.size(1)))
#             for f, proj_layer in zip(x, self.input_proj)
#         ]
#         memory = torch.cat(projected_feats, 1)  # (B, N_mem, hidden_dim)

#         dn_len = dn_meta["num_denoising_queries"]
#         det_len = dn_meta["num_matching_queries"]
#         dn_queries = q[:, :dn_len, :]
#         det_queries = q[:, dn_len:, :]

#         cls_all, box_all = [], []
#         for i in range(len(self.writer)):
#             # ---- SELF-ATTENTION: process groups independently for perfect symmetry ----
#             det_queries_ = self.writer[i](det_queries)
#             if dn_len > 0:
#                 dn_queries_ = self.writer[i](dn_queries)
#             else:
#                 dn_queries_ = dn_queries

#             # ---- CROSS-ATTENTION: process all together (memory is safe to share) ----
#             all_queries = torch.cat([dn_queries_, det_queries_], dim=1) if dn_len > 0 else det_queries_
#             all_queries = self.reader[i](memory, all_queries)

#             # ---- SPLIT for next layer ----
#             if dn_len > 0:
#                 dn_queries = all_queries[:, :dn_len, :]
#                 det_queries = all_queries[:, dn_len:, :]
#             else:
#                 det_queries = all_queries

#             # ---- Collect head outputs for each layer ----
#             head_idx = i if i < len(self.final_box_heads) else 0
#             cls_all.append(self.final_score_heads[head_idx](all_queries))
#             box_all.append(self.final_box_heads[head_idx](all_queries).sigmoid())
#         dec_scores, dec_boxes = torch.stack(cls_all), torch.stack(box_all)

#         k = dn_meta["num_matching_queries"]
#         if self.training:
#             return dec_boxes, dec_scores, enc_boxes[:, :k], enc_scores[:, :k], dn_meta
#         else:
#             if dn_meta["num_denoising_queries"] > 0:
#                 offset = dn_meta["num_denoising_queries"]
#                 out = torch.cat([dec_boxes[-1][:, offset:], dec_scores[-1][:, offset:].sigmoid()], -1)
#             else:
#                 out = torch.cat([dec_boxes[-1], dec_scores[-1].sigmoid()], -1)
#             dn_meta = None
#             return out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)


# class ViLBasedRTDETRHead(nn.Module):
#     def __init__(
#         self,
#         ch_vil,
#         vil_fpn_spatial_shapes,
#         hidden_dim=384,            # <== new fused dimension
#         num_queries=300,
#         num_vil_self_attn_layers=2,
#         vil_block_decoder_args=None,
#         act=nn.SiLU,
#         nc=80,
#         num_denoising=0,
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         grid_size_anchor=0.05,
#         return_intermediate_outputs=False,
#         fuse_out_size=40,         # <== fusion map size
#     ):
#         super().__init__()
#         self.nc, self.hidden_dim = int(nc), int(hidden_dim)
#         self.num_queries, self.num_denoising = int(num_queries), int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor
#         self.fuse_out_size = fuse_out_size

#         # Fusion layer: expects 4 PAFPN outputs with ch_vil[0] channels each (usually 192)
#         total_in_channels = sum(ch_vil)  # e.g., 256 + 256 + 256 = 768
#         self.fuse_to_single = nn.Conv2d(total_in_channels, self.hidden_dim, kernel_size=1)

#         # For single-scale mode, we use an Identity projection (no Linear needed)
#         self.input_proj = nn.ModuleList([nn.Identity()])

#         # Since we fuse to a single scale, we need just one spatial shape for anchors
#         self.register_buffer(
#             "vil_fpn_spatial_shapes",
#             torch.tensor([[fuse_out_size, fuse_out_size]], dtype=torch.long),
#             persistent=False
#         )

#         self.proposal_feature_transform = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
#         )
#         self.proposal_score_head = nn.Linear(hidden_dim, nc)
#         self.proposal_box_head   = MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU)

#         base_cfg = dict(
#             dim=hidden_dim,
#             expansion=2,
#             qkv_block_size=max(1, hidden_dim // 64 if hidden_dim >= 64 else hidden_dim // 16),
#             proj_bias=True,
#             norm_bias=True,
#             num_blocks=1,
#             gate_soft_cap=15.0,
#             chunk_size=64,
#         )
#         if vil_block_decoder_args is not None:
#             base_cfg.update(vil_block_decoder_args)
#         if hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next(
#                 (d for d in (16, 32, 64, 128, 256) if hidden_dim % d == 0), 1
#             )

#         self.writer = nn.ModuleList(
#             ViLLayerFusedQKV(
#                 **base_cfg,
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(num_vil_self_attn_layers)
#         )


#         directions = [
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT,    SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT,    SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#             (SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,   SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,   SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#         ]


#         self.reader = nn.ModuleList(
#             RefactoredxLSTMCrossReader(
#                 dim=hidden_dim,
#                 expansion=base_cfg["expansion"],
#                 qkv_block_size=base_cfg["qkv_block_size"],
#                 chunk_size=base_cfg["chunk_size"],
#                 gate_soft_cap=base_cfg["gate_soft_cap"],
#                 seqlens=[fuse_out_size, fuse_out_size],
#                 memory_direction=mem_dir,
#                 query_direction=q_dir,
#             )
#             for i, (mem_dir, q_dir) in enumerate(directions[:num_vil_self_attn_layers])
#         )

#         n_sets = max(1, num_vil_self_attn_layers) if return_intermediate_outputs else 1
#         self.final_score_heads = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(n_sets))
#         self.final_box_heads = nn.ModuleList(
#             MLP(hidden_dim, hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets)
#         )
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=nn.SiLU)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.weight.requires_grad:
#                 nn.init.xavier_uniform_(m.weight)
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#     @staticmethod
#     def _meshgrid(H, W, dtype, device):
#         y, x = torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device)
#         gy, gx = torch.meshgrid(y, x, indexing="ij")
#         return gx, gy

#     def _anchors(self, dtype, device):
#         shapes = self.vil_fpn_spatial_shapes.to(device)
#         out = []
#         for i, (H, W) in enumerate(shapes):
#             gx, gy = self._meshgrid(int(H), int(W), dtype, device)
#             cxcy = (torch.stack([gx, gy], -1).float() + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
#             wh   = torch.full_like(cxcy, self.grid_size_anchor * (2.0 ** i))
#             out.append(torch.cat([cxcy, wh], -1).view(-1, 4))
#         return torch.cat(out).unsqueeze(0)

#     def _initial_queries(self, feats, batch):
#         # feats: list with one fused feature map [B, hidden_dim, 16, 16]
#         proj_layer = self.input_proj[0]
#         f = feats[0]
#         # flatten: [B, hidden_dim, 16, 16] -> [B, 256, hidden_dim]
#         seq = proj_layer(f.permute(0,2,3,1).reshape(f.size(0), -1, f.size(1)))
#         cat = seq
#         bs, ntok, _ = cat.shape
#         anchors = self._anchors(cat.dtype, cat.device).repeat(bs, 1, 1)

#         dense = self.proposal_feature_transform(cat)
#         score = self.proposal_score_head(dense)
#         delta = self.proposal_box_head(dense)

#         ref_cxcy, ref_wh = anchors[..., :2], anchors[..., 2:]
#         boxes = torch.cat([ref_cxcy + delta[..., :2] * ref_wh,
#                            ref_wh * torch.exp(delta[..., 2:])], -1).sigmoid()

#         k = max(1, min(self.num_queries, ntok))
#         topk = score.sigmoid().amax(-1).topk(k, 1).indices
#         gather = lambda t, d: torch.gather(t, 1, topk.unsqueeze(-1).expand(-1, -1, d))
#         sel_feat, sel_box, sel_score = gather(cat, self.hidden_dim), gather(boxes, 4), gather(score, self.nc)

#         dn_meta = dict(
#             num_denoising_queries=0,
#             num_matching_queries=k,
#             dn_num_split=[0, k],
#             dn_pos_idx=[torch.zeros([0], dtype=torch.long, device=cat.device) for _ in range(bs)],
#             dn_num_group=0
#         )

#         dn_emb = dn_bbox = None
#         if self.training and self.num_denoising:
#             try:
#                 dn_emb, dn_bbox, _, extra = get_cdn_group(
#                     batch, self.nc, k, self.denoising_class_embed.weight,
#                     self.num_denoising, self.label_noise_ratio, self.box_noise_scale, True
#                 )
#                 dn_meta.update(extra)
#                 dn_meta["num_denoising_queries"] = dn_emb.size(1)
#                 dn_meta["dn_num_split"] = [dn_emb.size(1), k]
#             except Exception as e:
#                 warnings.warn(f"CDN disabled: {e}")

#         q_feat = torch.cat([dn_emb, sel_feat], 1) if dn_emb is not None else sel_feat
#         q_box = torch.cat([dn_bbox, sel_box], 1) if dn_bbox is not None else sel_box

#         if self.training:
#             q_feat = q_feat.detach()
#             q_box = q_box.detach()

#         return q_feat, q_box, sel_score, sel_box, dn_meta

#     def forward(self, x, batch=None):
#         # --- FUSION: PAFPN multi-scale -> single [B, 384, 16, 16] map ---
#         resized = [
#             F.adaptive_avg_pool2d(f, (self.fuse_out_size, self.fuse_out_size)) if f.shape[-1] > self.fuse_out_size
#             else F.interpolate(f, size=(self.fuse_out_size, self.fuse_out_size), mode='nearest') if f.shape[-1] < self.fuse_out_size
#             else f
#             for f in x
#         ]
#         fused = torch.cat(resized, dim=1)  # [B, 4*192, 16, 16]
#         fused = self.fuse_to_single(fused) # [B, 384, 16, 16]
#         x_fused = [fused]                  # single-map flow

#         # From here, everything is single-scale, just update as usual!
#         qf, qb, enc_scores, enc_boxes, dn_meta = (*self._initial_queries(x_fused, batch if self.training else None),)
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         q = qf.to(tgt_dtype) + self.query_pos_head(qb.to(tgt_dtype))

#         # Memory is from single 16x16 map (flattened)
#         memory = fused.permute(0,2,3,1).reshape(fused.size(0), -1, self.hidden_dim)  # [B, 256, 384]

#         dn_len = dn_meta["num_denoising_queries"]
#         det_len = dn_meta["num_matching_queries"]
#         dn_queries = q[:, :dn_len, :]
#         det_queries = q[:, dn_len:, :]

#         cls_all, box_all = [], []
#         for i in range(len(self.writer)):
#             det_queries_ = self.writer[i](det_queries)
#             if dn_len > 0:
#                 dn_queries_ = self.writer[i](dn_queries)
#             else:
#                 dn_queries_ = dn_queries

#             all_queries = torch.cat([dn_queries_, det_queries_], dim=1) if dn_len > 0 else det_queries_
#             all_queries = self.reader[i](memory, all_queries)

#             if dn_len > 0:
#                 dn_queries = all_queries[:, :dn_len, :]
#                 det_queries = all_queries[:, dn_len:, :]
#             else:
#                 det_queries = all_queries

#             head_idx = i if i < len(self.final_box_heads) else 0
#             cls_all.append(self.final_score_heads[head_idx](all_queries))
#             box_all.append(self.final_box_heads[head_idx](all_queries).sigmoid())
#         dec_scores, dec_boxes = torch.stack(cls_all), torch.stack(box_all)

#         k = dn_meta["num_matching_queries"]
#         if self.training:
#             return dec_boxes, dec_scores, enc_boxes[:, :k], enc_scores[:, :k], dn_meta
#         else:
#             if dn_meta["num_denoising_queries"] > 0:
#                 offset = dn_meta["num_denoising_queries"]
#                 out = torch.cat([dec_boxes[-1][:, offset:], dec_scores[-1][:, offset:].sigmoid()], -1)
#             else:
#                 out = torch.cat([dec_boxes[-1], dec_scores[-1].sigmoid()], -1)
#             dn_meta = None
#             return out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)



# class ViLBasedRTDETRHead(nn.Module):
#     def __init__(
#         self,
#         ch_vil,
#         vil_fpn_spatial_shapes,
#         hidden_dim=256,            # match RT-DETR
#         num_queries=300,
#         num_vil_self_attn_layers=6, # match RT-DETR’s 6 layers
#         vil_block_decoder_args=None,
#         act=nn.SiLU,
#         nc=80,
#         num_denoising=100,          # match RT-DETR default
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         grid_size_anchor=0.05,
#         return_intermediate_outputs=True,
#         fuse_out_size=40,
#     ):
#         super().__init__()
#         self.nc = int(nc)
#         self.hidden_dim = int(hidden_dim)
#         self.num_queries = int(num_queries)
#         self.num_denoising = int(num_denoising)
#         self.grid_size_anchor = grid_size_anchor
#         self.fuse_out_size = fuse_out_size

#         # --- Fusion from multi-scale PAFPN into one map ---
#         total_in_channels = sum(ch_vil)
#         self.fuse_to_single = nn.Conv2d(total_in_channels, self.hidden_dim, kernel_size=1, bias=True)

#         # Single‐scale identity projection
#         self.input_proj = nn.ModuleList([nn.Identity()])

#         # Register unified spatial shape
#         self.register_buffer(
#             "vil_fpn_spatial_shapes",
#             torch.tensor([[fuse_out_size, fuse_out_size]], dtype=torch.long),
#             persistent=False
#         )

#         # Proposal heads (encoder‐style)
#         self.proposal_feature_transform = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim)
#         )
#         self.proposal_score_head = nn.Linear(self.hidden_dim, self.nc)
#         self.proposal_box_head   = MLP(self.hidden_dim, self.hidden_dim, 4, 3, act=nn.SiLU)

#         # ViL‐based self‐attn (“writer”) blocks
#         base_cfg = dict(
#             dim=self.hidden_dim,
#             expansion=2,
#             qkv_block_size=max(1, self.hidden_dim // 64),
#             proj_bias=True,
#             norm_bias=True,
#             num_blocks=1,
#             gate_soft_cap=15.0,
#             chunk_size=64,
#         )
#         if vil_block_decoder_args is not None:
#             base_cfg.update(vil_block_decoder_args)
#         if self.hidden_dim % base_cfg["qkv_block_size"]:
#             base_cfg["qkv_block_size"] = next(
#                 (d for d in (16, 32, 64, 128, 256) if self.hidden_dim % d == 0), 1
#             )

#         self.writer = nn.ModuleList([
#             ViLLayerFusedQKV(
#                 **base_cfg,
#                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             )
#             for i in range(num_vil_self_attn_layers)
#         ])

#         # ViL‐based cross‐attn (“reader”) blocks with four traversal directions
#         directions = [
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT,  SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT,  SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#             (SequenceTraversal.ROWWISE_FROM_BOT_RIGHT, SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_BOT_RIGHT, SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#         ]
#         self.reader = nn.ModuleList([
#             RefactoredxLSTMCrossReader(
#                 dim=self.hidden_dim,
#                 expansion=base_cfg["expansion"],
#                 qkv_block_size=base_cfg["qkv_block_size"],
#                 chunk_size=base_cfg["chunk_size"],
#                 gate_soft_cap=base_cfg["gate_soft_cap"],
#                 seqlens=[fuse_out_size, fuse_out_size],
#                 memory_direction=mem_dir,
#                 query_direction=q_dir,
#             )
#             for i, (mem_dir, q_dir) in enumerate(directions[:num_vil_self_attn_layers])
#         ])

#         # One (score, box) head per layer for intermediate supervision
#         n_sets = num_vil_self_attn_layers
#         self.final_score_heads = nn.ModuleList(
#             nn.Linear(self.hidden_dim, self.nc) for _ in range(n_sets)
#         )
#         self.final_box_heads = nn.ModuleList(
#             MLP(self.hidden_dim, self.hidden_dim, 4, 3, act=nn.SiLU) for _ in range(n_sets)
#         )

#         # Denoising
#         self.denoising_class_embed = nn.Embedding(self.nc, self.hidden_dim) if self.num_denoising else None
#         self.label_noise_ratio, self.box_noise_scale = label_noise_ratio, box_noise_scale

#         # Positional encoding for queries
#         self.query_pos_head = MLP(4, 2 * self.hidden_dim, self.hidden_dim, 2, act=nn.SiLU)
        


#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Bias init
#         bias_cls = bias_init_with_prob(0.01) / 80 * self.nc

#         # Xavier‐uniform for all Linear and Conv layers
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.weight.requires_grad:
#                 nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     constant_(m.bias, 0.0)

#         # Heads
#         for h in self.final_score_heads:
#             constant_(h.bias, bias_cls)
#         for m in self.final_box_heads:
#             constant_(m.layers[-1].weight, 0.0)
#             constant_(m.layers[-1].bias,   0.0)

#         # Proposal heads & query-pos head (already covered by modules loop,
#         # but ensure biases zeroed)
#         for m in self.proposal_feature_transform:
#             if isinstance(m, nn.Linear):
#                 constant_(m.bias, 0.0)
#         for layer in self.query_pos_head.layers:
#             if isinstance(layer, nn.Linear):
#                 constant_(layer.bias, 0.0)

#     @staticmethod
#     def _meshgrid(H, W, dtype, device):
#         y = torch.arange(H, dtype=dtype, device=device)
#         x = torch.arange(W, dtype=dtype, device=device)
#         return torch.meshgrid(y, x, indexing="ij")

#     def _anchors(self, dtype, device, eps=1e-2):
#         # Logit‐space anchors + mask (optional)
#         shapes = self.vil_fpn_spatial_shapes.to(device)
#         boxes = []
#         for i, (H, W) in enumerate(shapes):
#             gy, gx = self._meshgrid(H, W, dtype, device)
#             grid = torch.stack([gx, gy], -1).float()
#             grid = (grid + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
#             wh = torch.full_like(grid, self.grid_size_anchor * (2.0**i))
#             box = torch.cat([grid, wh], -1).view(-1, 4)  # normalized [0,1]
#             boxes.append(box)
#         anchors = torch.cat(boxes, dim=0).unsqueeze(0)        # (1, N, 4)
#         anchors = torch.clamp(anchors, eps, 1 - eps)
#         return torch.log(anchors / (1 - anchors))            # (1, N, 4) logits

#     def _initial_queries(self, feats, batch):
#         # feats: single fused map → [B, hidden_dim, S, S]
#         proj = self.input_proj[0]
#         f = feats[0]
#         seq = proj(f.permute(0,2,3,1).reshape(f.size(0), -1, f.size(1)))
#         bs, ntok, _ = seq.shape

#         anchors = self._anchors(seq.dtype, seq.device).repeat(bs, 1, 1)
#         dense = self.proposal_feature_transform(seq)
#         score = self.proposal_score_head(dense)
#         delta = self.proposal_box_head(dense)

#         ref_cxcy, ref_wh = anchors[..., :2], anchors[..., 2:]
#         boxes = torch.cat([
#             ref_cxcy + delta[..., :2] * ref_wh,
#             ref_wh * torch.exp(delta[..., 2:])
#         ], -1).sigmoid()

#         k = max(1, min(self.num_queries, ntok))
#         topk = score.sigmoid().amax(-1).topk(k, dim=1).indices
#         gather = lambda t, d: torch.gather(t, 1, topk.unsqueeze(-1).expand(-1, -1, d))

#         sel_feat, sel_box, sel_score = (
#             gather(seq, self.hidden_dim),
#             gather(boxes, 4),
#             gather(score, self.nc),
#         )

#         # --- Denoising (no try/except) ---
#         dn_meta = dict(
#             num_denoising_queries=0,
#             num_matching_queries=k,
#             dn_num_split=[0, k],
#             dn_pos_idx=[torch.zeros([0], dtype=torch.long, device=seq.device) for _ in range(bs)],
#             dn_num_group=0,
#         )
#         dn_emb = dn_bbox = None
#         if self.training and self.num_denoising:
#             dn_emb, dn_bbox, _, extra = get_cdn_group(
#                 batch, self.nc, k,
#                 self.denoising_class_embed.weight,
#                 self.num_denoising,
#                 self.label_noise_ratio,
#                 self.box_noise_scale,
#                 True
#             )
#             dn_meta.update(extra)
#             dn_meta["num_denoising_queries"] = dn_emb.size(1)
#             dn_meta["dn_num_split"] = [dn_emb.size(1), k]

#         # Detach training queries
#         q_feat = torch.cat([dn_emb, sel_feat], 1) if dn_emb is not None else sel_feat
#         q_box  = torch.cat([dn_bbox, sel_box], 1) if dn_bbox is not None else sel_box
#         if self.training:
#             q_feat = q_feat.detach()
#             q_box  = q_box.detach()

#         return q_feat, q_box, sel_score, sel_box, dn_meta

#     def forward(self, x, batch=None):
#         # --- Fuse multi-scale PAFPN → single map ---
#         resized = []
#         for f in x:
#             if f.shape[-1] > self.fuse_out_size:
#                 resized.append(F.adaptive_avg_pool2d(f, (self.fuse_out_size,) * 2))
#             elif f.shape[-1] < self.fuse_out_size:
#                 resized.append(F.interpolate(f, size=(self.fuse_out_size,) * 2, mode='nearest'))
#             else:
#                 resized.append(f)
#         fused = torch.cat(resized, dim=1)
#         fused = self.fuse_to_single(fused)  # [B, hidden_dim, S, S]
#         x_fused = [fused]

#         # --- Queries & denoising setup ---
#         qf, qb, enc_scores, enc_boxes, dn_meta = self._initial_queries(x_fused, batch if self.training else None)
#         tgt_dtype = self.query_pos_head.layers[0].weight.dtype
#         q = qf.to(tgt_dtype) + self.query_pos_head(qb.to(tgt_dtype))

#         # Flatten memory
#         memory = fused.permute(0,2,3,1).reshape(fused.size(0), -1, self.hidden_dim)

#         dn_len = dn_meta["num_denoising_queries"]
#         det_len = dn_meta["num_matching_queries"]
#         dn_queries = q[:, :dn_len, :]
#         det_queries = q[:, dn_len:, :]

#         cls_all, box_all = [], []
#         for i, writer in enumerate(self.writer):
#             det_q_ = writer(det_queries)
#             dn_q_  = writer(dn_queries) if dn_len > 0 else dn_queries
#             all_q  = torch.cat([dn_q_, det_q_], dim=1) if dn_len > 0 else det_q_
#             all_q  = self.reader[i](memory, all_q)

#             if dn_len > 0:
#                 dn_queries = all_q[:, :dn_len, :]
#                 det_queries = all_q[:, dn_len:, :]
#             else:
#                 det_queries = all_q

#             head_idx = i
#             cls_all.append(self.final_score_heads[head_idx](all_q))
#             box_all.append(self.final_box_heads[head_idx](all_q).sigmoid())

#         dec_scores = torch.stack(cls_all)
#         dec_boxes  = torch.stack(box_all)

#         if self.training:
#             return dec_boxes, dec_scores, enc_boxes[:, :det_len], enc_scores[:, :det_len], dn_meta

#         # Inference: drop denoising
#         if dn_len > 0:
#             offset = dn_len
#             out = torch.cat([
#                 dec_boxes[-1][:, offset:],
#                 dec_scores[-1][:, offset:].sigmoid()
#             ], -1)
#         else:
#             out = torch.cat([dec_boxes[-1], dec_scores[-1].sigmoid()], -1)

#         return (out, (dec_boxes, dec_scores, enc_boxes, enc_scores, None))





# class ViLBasedRTDETRHead(nn.Module):
#     def __init__(
#         self,
#         ch_vil, # Channels from PAFPN for fusion
#         # vil_fpn_spatial_shapes, # This will be determined by fuse_out_size
#         hidden_dim=256,         # Matches RT-DETR 'hd'
#         num_queries=300,        # Matches RT-DETR 'nq'
#         num_vil_layers=6,       # Matches RT-DETR 'ndl' (formerly num_vil_self_attn_layers)
#         vil_block_decoder_args=None, # Keep for ViL blocks
#         act=nn.SiLU,            # Activation for MLPs (RT-DETR uses ReLU in its MLP by default, but can be changed)
#                                 # Note: Your ViL blocks use SiLU internally. RT-DETR's deformable layers also often use SiLU/GeLU
#         nc=80,
#         # Denoising parameters (matching RT-DETR naming/defaults)
#         num_denoising=100,      # Matches RT-DETR 'nd'
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         # Anchors
#         grid_size_anchor=0.05,  # Keep this, as it's for your single fused scale
#         # Query Embeddings
#         learnt_init_query=False, # Matches RT-DETR, make it an option
#         # MLP for query_pos_head (matching RT-DETR structure)
#         query_pos_head_num_layers=2,
#         # MLP for bbox heads (matching RT-DETR structure)
#         bbox_head_num_layers=3,
#         return_intermediate_outputs=True, # Usually True during training for DETRs
#         fuse_out_size=40,       # Example: Fused map size (e.g., S/32 if S=640)
#     ):
#         super().__init__()
#         self.nc = int(nc)
#         self.hidden_dim = int(hidden_dim)
#         self.num_queries = int(num_queries)
#         self.num_vil_layers = int(num_vil_layers) # num_decoder_layers
#         self.return_intermediate_outputs = return_intermediate_outputs
        

#         self.grid_size_anchor = grid_size_anchor
#         self.fuse_out_size = fuse_out_size

#         # --- Feature Fusion (Your Specific Part) ---
#         # This part is unique to your HEAD 1 and is kept as per your request.
#         total_in_channels = sum(ch_vil)
#         self.fuse_to_single = nn.Conv2d(total_in_channels, self.hidden_dim, kernel_size=1)
#         # -------------------------------------------

#         # Input projection for multi-scale features is NOT used here,
#         # as we fuse first. self.input_proj = nn.ModuleList([nn.Identity()]) from your original code
#         # seems okay if ViL blocks expect a list, but if they expect a tensor, this might change.
#         # For _initial_queries, we will operate on the single fused map directly.
#         # So, self.input_proj isn't strictly needed in the RT-DETR sense.

#         # Spatial shapes for anchors: always the single fused shape
#         self.register_buffer(
#             "vil_fpn_spatial_shapes", # Name kept for consistency with your code
#             torch.tensor([[fuse_out_size, fuse_out_size]], dtype=torch.long),
#             persistent=False
#         )

#         # --- Encoder Head (for initial query selection from fused map) ---
#         # Mimicking RT-DETR's self.enc_output, self.enc_score_head, self.enc_bbox_head
#         self.enc_output = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )
#         self.enc_score_head = nn.Linear(hidden_dim, nc)
#         self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=bbox_head_num_layers, act=nn.SiLU) # Matched structure

#         # --- Denoising Part (Matching RT-DETR) ---
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if num_denoising > 0 else None
#         self.num_denoising = num_denoising
#         self.label_noise_ratio = label_noise_ratio
#         self.box_noise_scale = box_noise_scale

#         # --- Decoder/Query Embedding (Matching RT-DETR) ---
#         self.learnt_init_query = learnt_init_query
#         if learnt_init_query:
#             self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
#         # Positional encoding from boxes
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=query_pos_head_num_layers, act=nn.SiLU) # Matched structure


#         # --- ViL Blocks (Your Specific Part - "Writer" and "Reader") ---
#         # This part is unique to your HEAD 1 and is kept.
#         # Assuming base_cfg and directions are defined correctly for your ViL blocks
#         base_cfg = dict( # Example, adjust as needed
#             dim=hidden_dim, expansion=2, qkv_block_size=max(1,hidden_dim // 64),
#             # ... other ViL specific params
#         )
#         if vil_block_decoder_args is not None:
#              base_cfg.update(vil_block_decoder_args)

#         self.writer = nn.ModuleList(
#             ViLLayerFusedQKV(
#                  **base_cfg,
#                  direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(self.num_vil_layers)
#         )
#         vil_directions = [ # Example, adjust as needed for RefactoredxLSTMCrossReader
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT, SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT, SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#         ] * ( (self.num_vil_layers + 1) // 2)

#         self.reader = nn.ModuleList(
#             RefactoredxLSTMCrossReader(
#                 dim=hidden_dim, seqlens=[fuse_out_size, fuse_out_size], # Pass fused size
#                 # ... other RefactoredxLSTMCrossReader specific params from base_cfg ...
#                 expansion=base_cfg["expansion"], qkv_block_size=base_cfg["qkv_block_size"],
#                 memory_direction=mem_dir, query_direction=q_dir,
#             ) for i, (mem_dir, q_dir) in enumerate(vil_directions[:self.num_vil_layers])
#         )
#         # -------------------------------------------------------------

#         # --- Final Prediction Heads (Matching RT-DETR structure) ---
#         # Renamed from final_score_heads / final_box_heads
#         # One set of heads per ViL layer output if return_intermediate_outputs
#         num_pred_heads = self.num_vil_layers if self.return_intermediate_outputs else 1

#         self.dec_score_head = nn.ModuleList(
#             [nn.Linear(hidden_dim, nc) for _ in range(num_pred_heads)]
#         )
#         self.dec_bbox_head = nn.ModuleList(
#             [MLP(hidden_dim, hidden_dim, 4, num_layers=bbox_head_num_layers, act=nn.SiLU) for _ in range(num_pred_heads)]
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Adopt RT-DETR's initialization for matching components
#         bias_cls = bias_init_with_prob(0.01) # RT-DETR uses 0.01
#         # if self.nc != 80: # Ultralytics RT-DETR adjusts this if nc is not 80.
#         #    bias_cls = bias_cls / 80 * self.nc # Optional adjustment

#         # Encoder head
#         if hasattr(self.enc_output[0], 'weight'): # Linear layer
#             xavier_uniform_(self.enc_output[0].weight)
#         # constant_(self.enc_score_head.bias, bias_cls) # Done in loop below for all score heads
#         if hasattr(self.enc_bbox_head.layers[-1], 'weight'): # Last layer of MLP
#             constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
#             constant_(self.enc_bbox_head.layers[-1].bias, 0.0)

#         # Decoder/Query Embeddings
#         if self.learnt_init_query:
#             xavier_uniform_(self.tgt_embed.weight)
#         if hasattr(self.query_pos_head.layers[0], 'weight'): # First layer of MLP
#             xavier_uniform_(self.query_pos_head.layers[0].weight)
#         if len(self.query_pos_head.layers) > 1 and hasattr(self.query_pos_head.layers[1], 'weight'): # Second layer
#              xavier_uniform_(self.query_pos_head.layers[1].weight)


#         # Final prediction heads (dec_score_head, dec_bbox_head)
#         for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
#             constant_(cls_.bias, bias_cls)
#             if hasattr(reg_.layers[-1], 'weight'): # Last layer of MLP
#                 constant_(reg_.layers[-1].weight, 0.0)
#                 constant_(reg_.layers[-1].bias, 0.0)
        
#         # Also initialize enc_score_head bias (if not covered by dec_score_head loop, e.g. if not return_intermediate)
#         constant_(self.enc_score_head.bias, bias_cls)


#         # Initialize other ViL-specific layers (example, adjust as needed)
#         # Your original code initializes all nn.Linear with xavier_uniform_.
#         # This might be okay for the ViL blocks if they don't have specific needs.
#         # Or, ViLLayerFusedQKV and RefactoredxLSTMCrossReader should have their own reset_parameters method.
#         for m in self.writer.modules():
#             if hasattr(m, 'reset_parameters') and m != self: m.reset_parameters()
#         for m in self.reader.modules():
#             if hasattr(m, 'reset_parameters') and m != self: m.reset_parameters()
#         if hasattr(self.fuse_to_single, 'weight'):
#              xavier_uniform_(self.fuse_to_single.weight)
#              if self.fuse_to_single.bias is not None:
#                  constant_(self.fuse_to_single.bias, 0.0)


#     def _generate_anchors(self, B, dtype, device, eps=1e-2): # Matched signature from RT-DETR
#         # Generates anchors for the single fused scale, but applies logit transformation
#         # Matching RT-DETR's _generate_anchors structure, adapted for single scale
#         shapes = self.vil_fpn_spatial_shapes.to(device) # Should be [[fuse_out_size, fuse_out_size]]
#         anchors = []
#         # i will only be 0
#         for i, (H_fused, W_fused) in enumerate(shapes):
#             # Meshgrid (gx, gy)
#             sy = torch.arange(end=H_fused.item(), dtype=dtype, device=device)
#             sx = torch.arange(end=W_fused.item(), dtype=dtype, device=device)
#             # Consider torch.meshgrid indexing if needed for older PyTorch
#             grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
#             grid_xy = torch.stack([grid_x, grid_y], -1)  # [H_fused, W_fused, 2]

#             valid_WH = torch.tensor([W_fused, H_fused], dtype=dtype, device=device)
#             # Normalized cx, cy
#             grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH # [1, H_fused, W_fused, 2]
#             # Anchor wh
#             # In RT-DETR: grid_size * (2.0**i). Here i is always 0 for the single fused scale.
#             wh = torch.ones_like(grid_xy) * self.grid_size_anchor # [1, H_fused, W_fused, 2]
            
#             # Concatenate cxcywh and flatten
#             current_anchors = torch.cat([grid_xy, wh], -1).view(1, -1, 4) # [1, H_fused*W_fused, 4]
#             anchors.append(current_anchors)
        
#         anchors = torch.cat(anchors, 1) # [1, num_total_anchors, 4] (num_total_anchors = H_fused*W_fused)

#         # Apply valid_mask and logit transformation like in RT-DETR
#         valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
#         anchors = torch.log(anchors / (1 - anchors)) # To logit space
#         anchors = anchors.masked_fill(~valid_mask, float("inf"))
        
#         return anchors.repeat(B, 1, 1), valid_mask.repeat(B, 1, 1) # Repeat for batch

#     def _initial_queries(self, fused_map, batch): # Fused map is [B, hidden_dim, H_fused, W_fused]
#             B, C, H_fused, W_fused = fused_map.shape
            
#             # Flatten: [B, H_fused*W_fused, hidden_dim]
#             flattened_features = fused_map.flatten(2).permute(0, 2, 1)

#             # Generate anchors (now in logit space)
#             anchors_logit, valid_mask = self._generate_anchors(B, flattened_features.dtype, flattened_features.device)
            
#             # Apply encoder heads to flattened features
#             transformed_features = self.enc_output(valid_mask * flattened_features)

#             scores_logits = self.enc_score_head(transformed_features)
#             boxes_delta_logit = self.enc_bbox_head(transformed_features)
            
#             initial_boxes_logit = boxes_delta_logit + anchors_logit
#             initial_boxes_sigmoid = initial_boxes_logit.sigmoid()

#             ntok = flattened_features.shape[1]
#             actual_k = min(self.num_queries, ntok)
            
#             topk_scores, topk_indices = torch.topk(scores_logits.max(-1).values, actual_k, dim=1)

#             expanded_topk_indices_feat = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
#             sel_feat_content = torch.gather(transformed_features, 1, expanded_topk_indices_feat)
            
#             expanded_topk_indices_box = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
#             sel_box_sigmoid = torch.gather(initial_boxes_sigmoid, 1, expanded_topk_indices_box)
#             sel_box_logit = torch.gather(initial_boxes_logit, 1, expanded_topk_indices_box)

#             expanded_topk_indices_score = topk_indices.unsqueeze(-1).expand(-1, -1, self.nc)
#             sel_score_logits = torch.gather(scores_logits, 1, expanded_topk_indices_score)

#             # --- Denoising ---
#             dn_embed_content, dn_bbox_logit, attn_mask, dn_meta = get_cdn_group(
#                 batch, self.nc, actual_k,
#                 self.denoising_class_embed.weight if self.denoising_class_embed else None,
#                 self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training
#             )

#             # Prepare query features and reference boxes for ViL decoder
#             if self.learnt_init_query:
#                 query_content_features_base = self.tgt_embed.weight.unsqueeze(0).repeat(B, 1, 1)
#             else:
#                 query_content_features_base = sel_feat_content

#             reference_boxes_logit_base = sel_box_logit

#             # IMPORTANT: Detach if in training mode, and not learned queries
#             if self.training:
#                 if not self.learnt_init_query: # Only detach if not learnt
#                     query_content_features_base = query_content_features_base.detach()
#                 reference_boxes_logit_base = reference_boxes_logit_base.detach()
#                 if dn_embed_content is not None: # Only detach if denoising is actually present
#                     dn_embed_content = dn_embed_content.detach()
#                 if dn_bbox_logit is not None: # Only detach if denoising is actually present
#                     dn_bbox_logit = dn_bbox_logit.detach()


#             # Concatenate denoising queries for the decoder input
#             # Check if dn_embed_content and dn_bbox_logit are not None,
#             # which implies denoising is active and returned valid tensors.
#             if dn_embed_content is not None and dn_bbox_logit is not None:
#                 q_content_for_decoder = torch.cat([dn_embed_content, query_content_features_base], dim=1)
#                 q_box_logit_for_decoder = torch.cat([dn_bbox_logit, reference_boxes_logit_base], dim=1)
#             else:
#                 q_content_for_decoder = query_content_features_base
#                 q_box_logit_for_decoder = reference_boxes_logit_base
#                 # If no denoising, ensure dn_meta is also correctly indicating no denoising
#                 if dn_meta is not None: # dn_meta could be None if self.training is False
#                     dn_meta['num_denoising_queries'] = 0
#                     dn_meta['dn_num_split'][0] = 0
#                 # If dn_meta is None (e.g., self.training is False), no need to set anything

#             # --- Fix for KeyError: 'num_denoising_queries' ---
#             # Get the actual number of denoising queries from dn_meta.
#             # Only access dn_meta if it's not None and we are in training.
#             current_num_denoising_queries = 0
#             if self.training and dn_meta is not None and 'num_denoising_queries' in dn_meta:
#                 current_num_denoising_queries = dn_meta['num_denoising_queries']
#             # --- End Fix ---

#             if self.training and current_num_denoising_queries > 0:
#                 # Create `enc_boxes_for_loss` that includes denoising part, converted to sigmoid
#                 enc_boxes_for_loss = torch.cat([dn_bbox_logit.sigmoid(), sel_box_sigmoid], dim=1)

#                 placeholder_dn_scores = torch.full(
#                     (B, current_num_denoising_queries, self.nc),
#                     0.0, # Placeholder value
#                     device=sel_score_logits.device,
#                     dtype=sel_score_logits.dtype
#                 )
#                 enc_scores_for_loss = torch.cat([placeholder_dn_scores, sel_score_logits], dim=1)
#             else:
#                 enc_boxes_for_loss = sel_box_sigmoid
#                 enc_scores_for_loss = sel_score_logits

#             return q_content_for_decoder, q_box_logit_for_decoder, enc_boxes_for_loss, enc_scores_for_loss, dn_meta


#     def forward(self, x, batch=None): # x is list of PAFPN outputs
#         # --- FUSION: PAFPN multi-scale -> single [B, hidden_dim, H_fused, W_fused] map ---
#         # This is your specific part
#         resized = [
#             F.adaptive_avg_pool2d(f, (self.fuse_out_size, self.fuse_out_size)) if f.shape[-1] > self.fuse_out_size
#             else F.interpolate(f, size=(self.fuse_out_size, self.fuse_out_size), mode='nearest') if f.shape[-1] < self.fuse_out_size
#             else f
#             for f_idx, f in enumerate(x) # x is assumed to be a list of features from backbone
#         ]
#         fused_cat = torch.cat(resized, dim=1)
#         fused_map = self.fuse_to_single(fused_cat) # [B, hidden_dim, H_fused, W_fused]
#         # -------------------------------------------

#         # Initial queries from the fused map (encoder proposals)
#         # q_content, q_box_logit, enc_boxes_sigmoid, enc_scores_logits, dn_meta
#         q_content, qb_logit, enc_boxes, enc_scores, dn_meta = self._initial_queries(fused_map, batch)
        
#         # Add positional embedding from boxes to content features
#         # qb_logit is [B, N_queries_total, 4]
#         # query_pos_head expects normalized boxes, but RT-DETR passes logit boxes and hopes MLP learns.
#         # Let's pass sigmoided boxes to query_pos_head if it expects normalized [0,1]
#         # Or, ensure query_pos_head can handle logit.
#         # Given RT-DETR structure, query_pos_head is applied to reference points (logit or sigmoid)
#         # If query_pos_head is like RT-DETR's, it expects 4D input.
#         # Let's assume it works fine on logit_boxes, or sigmoid(logit_boxes)
#         # Common practice is to use sigmoid(logit_boxes) for the positional embedding generation.
#         pos_embed = self.query_pos_head(qb_logit.sigmoid()) # Or qb_logit directly if MLP handles it
#         q_final_content = q_content + pos_embed # These are the queries for ViL blocks

#         # Memory for ViL blocks (from the single fused map)
#         memory = fused_map.permute(0,2,3,1).reshape(fused_map.size(0), -1, self.hidden_dim)

#         # --- ViL Decoder Iterations (Your Specific Part) ---
#         # dn_len and det_len from dn_meta
#         dn_len = dn_meta['num_denoising_queries'] if dn_meta and 'num_denoising_queries' in dn_meta else 0
#         # det_len = dn_meta['num_matching_queries'] if dn_meta else q_final_content.size(1) - dn_len
        
#         # q_final_content is now [B, total_queries, C]
#         # Split into dn_queries and det_queries for ViL processing if ViL needs it
#         current_queries = q_final_content # This is `tgt` or `embed` in RT-DETR decoder

#         all_layer_outputs_scores_logits = []
#         all_layer_outputs_boxes_sigmoid = []

#         # Loop through ViL layers
#         for i in range(self.num_vil_layers):
#             # Queries might be modified by self-attention (writer)
#             # Then cross-attention with memory (reader)
#             # This logic needs to fit how your ViL writer/reader expect inputs
#             # Assuming writer (self-attn like) then reader (cross-attn like)
#             # Example flow:
#             # current_queries_self_attended = self.writer[i](current_queries) # If writer takes all queries
#             # current_queries_cross_attended = self.reader[i](memory, current_queries_self_attended)
#             # For simplicity, if your writer+reader form one "decoder layer":
            
#             # Simplified: Assume writer does self-attention part, reader does cross-attention part
#             # Need to handle dn_queries and det_queries separately if ViL blocks require specific processing
#             # like some transformer decoders do (e.g. only cross-attend for detection queries).
#             # For now, treat all queries together for ViL blocks:
            
#             temp_queries = self.writer[i](current_queries)
#             current_queries = self.reader[i](memory, temp_queries) # Output of ViL layer [B, N_queries_total, C]

#             # Get predictions from this layer's heads
#             # Head index for intermediate outputs
#             head_idx = i if self.return_intermediate_outputs else 0
            
#             layer_scores_logits = self.dec_score_head[head_idx](current_queries) # [B, N_queries_total, nc]
            
#             # Box prediction: RT-DETR adds deltas to reference points.
#             # Here, qb_logit are the reference points from the previous stage (or encoder).
#             # So, dec_bbox_head should predict deltas for these.
#             layer_boxes_delta_logit = self.dec_bbox_head[head_idx](current_queries) # [B, N_queries_total, 4]
#             layer_boxes_logit = qb_logit + layer_boxes_delta_logit # Add delta to reference boxes
#             layer_boxes_sigmoid = layer_boxes_logit.sigmoid()

#             all_layer_outputs_scores_logits.append(layer_scores_logits)
#             all_layer_outputs_boxes_sigmoid.append(layer_boxes_sigmoid)
            
#             # Update qb_logit for the next layer if iterative refinement of boxes is desired as reference
#             if self.training: # Detach for next layer's reference if iterative refinement
#                 qb_logit = layer_boxes_logit.detach()
#             else:
#                 qb_logit = layer_boxes_logit
        
#         # Stack intermediate outputs (matching RT-DETR output format for decoder)
#         # [num_layers, B, N_queries_total, C]
#         dec_scores_logits = torch.stack(all_layer_outputs_scores_logits)
#         dec_boxes_sigmoid = torch.stack(all_layer_outputs_boxes_sigmoid)

#         # Prepare final output (matching RT-DETR)
#         # enc_boxes and enc_scores are from the initial_queries stage
#         # dec_boxes_sigmoid and dec_scores_logits are from the ViL decoder stages
        
#         # k_enc = enc_scores.size(1) # Number of proposals from encoder stage
#         # For loss calculation, enc_scores and enc_boxes are needed.
#         # RT-DETR might select top-k from enc_scores/enc_boxes for loss if different from num_decoder_queries
#         # Here, enc_scores/enc_boxes from _initial_queries are already top-k based on num_queries
        
#         if self.training:
#             return dec_boxes_sigmoid, dec_scores_logits, enc_boxes, enc_scores, dn_meta
#         else:
#             # Inference: use outputs from the last ViL layer
#             last_dec_boxes_sigmoid = dec_boxes_sigmoid[-1] # [B, N_queries_total, 4]
#             last_dec_scores_logits = dec_scores_logits[-1] # [B, N_queries_total, nc]

#             # Remove denoising queries for final output if they exist
#             if dn_meta and dn_meta.get('num_denoising_queries', 0) > 0:
#                 dn_queries_count = dn_meta['num_denoising_queries']
#                 # Output only detection queries
#                 final_boxes = last_dec_boxes_sigmoid[:, dn_queries_count:, :]
#                 final_scores = last_dec_scores_logits[:, dn_queries_count:, :].sigmoid() # Apply sigmoid for scores
#             else:
#                 final_boxes = last_dec_boxes_sigmoid
#                 final_scores = last_dec_scores_logits.sigmoid()

#             # Concatenate boxes and scores: [B, N_det_queries, 4+nc]
#             # This matches Ultralytics RT-DETR's inference output format
#             # Note: RT-DETR output for export might be squeezed if B=1
#             # y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
#             # Here, final_boxes/final_scores are already [B, N, Dim]
#             output = torch.cat([final_boxes, final_scores], dim=-1)
            
#             # Original RT-DETR returns (y, (all_outputs_for_eval_if_needed))
#             # For simplicity, just return concatenated output, or adapt as needed.
#             all_outputs_tuple = (dec_boxes_sigmoid, dec_scores_logits, enc_boxes, enc_scores, dn_meta)
#             return output, all_outputs_tuple # Or just output


#10 MAP, topping out there.
# class ViLBasedRTDETRHead(nn.Module):
#     def __init__(
#         self,
#         ch_vil, # Channels from PAFPN for fusion
#         # vil_fpn_spatial_shapes, # This will be determined by fuse_out_size
#         hidden_dim=256,         # Matches RT-DETR 'hd'
#         num_queries=300,        # Matches RT-DETR 'nq'
#         num_vil_layers=6,       # Matches RT-DETR 'ndl' (formerly num_vil_self_attn_layers)
#         vil_block_decoder_args=None, # Keep for ViL blocks
#         act=nn.SiLU,            # Activation for MLPs (RT-DETR uses ReLU in its MLP by default, but can be changed)
#                                 # Note: Your ViL blocks use SiLU internally. RT-DETR's deformable layers also often use SiLU/GeLU
#         nc=80,
#         # Denoising parameters (matching RT-DETR naming/defaults)
#         num_denoising=100,      # Matches RT-DETR 'nd'
#         label_noise_ratio=0.5,
#         box_noise_scale=1.0,
#         # Anchors
#         grid_size_anchor=0.05,  # Keep this, as it's for your single fused scale
#         # Query Embeddings
#         learnt_init_query=False, # Matches RT-DETR, make it an option
#         # MLP for query_pos_head (matching RT-DETR structure)
#         query_pos_head_num_layers=2,
#         # MLP for bbox heads (matching RT-DETR structure)
#         bbox_head_num_layers=3,
#         return_intermediate_outputs=True, # Usually True during training for DETRs
#         fuse_out_size=40,       # Example: Fused map size (e.g., S/32 if S=640)
#     ):
#         super().__init__()
#         self.nc = int(nc)
#         self.hidden_dim = int(hidden_dim)
#         self.num_queries = int(num_queries)
#         self.num_vil_layers = int(num_vil_layers) # num_decoder_layers
#         self.return_intermediate_outputs = return_intermediate_outputs



#         self.grid_size_anchor = grid_size_anchor
#         self.fuse_out_size = fuse_out_size

#         # --- Feature Fusion (Your Specific Part) ---
#         # This part is unique to your HEAD 1 and is kept as per your request.
#         total_in_channels = sum(ch_vil)
#         self.fuse_to_single = nn.Conv2d(total_in_channels, self.hidden_dim, kernel_size=1)
#         self.fuse_norm = nn.RMSNorm(self.hidden_dim, eps=1e-6, elementwise_affine=True)
#         # -------------------------------------------

#         # Input projection for multi-scale features is NOT used here,
#         # as we fuse first. self.input_proj = nn.ModuleList([nn.Identity()]) from your original code
#         # seems okay if ViL blocks expect a list, but if they expect a tensor, this might change.
#         # For _initial_queries, we will operate on the single fused map directly.
#         # So, self.input_proj isn't strictly needed in the RT-DETR sense.

#         # Spatial shapes for anchors: always the single fused shape
#         self.register_buffer(
#             "vil_fpn_spatial_shapes", # Name kept for consistency with your code
#             torch.tensor([[fuse_out_size, fuse_out_size]], dtype=torch.long),
#             persistent=False
#         )

        

#         # --- Encoder Head (for initial query selection from fused map) ---
#         # Mimicking RT-DETR's self.enc_output, self.enc_score_head, self.enc_bbox_head
#         self.enc_output = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )
#         self.enc_score_head = nn.Linear(hidden_dim, nc)
#         self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=bbox_head_num_layers, act=nn.SiLU) # Matched structure

#         # --- Denoising Part (Matching RT-DETR) ---
#         self.denoising_class_embed = nn.Embedding(nc, hidden_dim) if num_denoising > 0 else None
#         self.num_denoising = num_denoising
#         self.label_noise_ratio = label_noise_ratio
#         self.box_noise_scale = box_noise_scale

#         # --- Decoder/Query Embedding (Matching RT-DETR) ---
#         self.learnt_init_query = learnt_init_query
#         if learnt_init_query:
#             self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
#         # Positional encoding from boxes
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=query_pos_head_num_layers, act=nn.SiLU) # Matched structure

#         #RMSNorm Decoder Queries
#         self.decoder_norm = nn.RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=True)

#         # --- ViL Blocks (Your Specific Part - "Writer" and "Reader") ---
#         # This part is unique to your HEAD 1 and is kept.
#         # Assuming base_cfg and directions are defined correctly for your ViL blocks
#         base_cfg = dict( # Example, adjust as needed
#             dim=hidden_dim, expansion=2, qkv_block_size=max(1,hidden_dim // 64),
#             # ... other ViL specific params
#         )
#         if vil_block_decoder_args is not None:
#              base_cfg.update(vil_block_decoder_args)

#         self.writer = nn.ModuleList(
#             ViLLayerFusedQKV(
#                  **base_cfg,
#                  direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
#             ) for i in range(self.num_vil_layers)
#         )
#         vil_directions = [ # Example, adjust as needed for RefactoredxLSTMCrossReader
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT, SequenceTraversal.ROWWISE_FROM_TOP_LEFT),
#             (SequenceTraversal.ROWWISE_FROM_TOP_LEFT, SequenceTraversal.ROWWISE_FROM_BOT_RIGHT),
#         ] * ( (self.num_vil_layers + 1) // 2)

#         self.reader = nn.ModuleList(
#             RefactoredxLSTMCrossReader(
#                 dim=hidden_dim, seqlens=[fuse_out_size, fuse_out_size], # Pass fused size
#                 # ... other RefactoredxLSTMCrossReader specific params from base_cfg ...
#                 expansion=base_cfg["expansion"], qkv_block_size=base_cfg["qkv_block_size"],
#                 memory_direction=mem_dir, query_direction=q_dir,
#             ) for i, (mem_dir, q_dir) in enumerate(vil_directions[:self.num_vil_layers])
#         )
#         # -------------------------------------------------------------

#         # --- Final Prediction Heads (Matching RT-DETR structure) ---
#         # Renamed from final_score_heads / final_box_heads
#         # One set of heads per ViL layer output if return_intermediate_outputs
#         num_pred_heads = self.num_vil_layers if self.return_intermediate_outputs else 1

#         self.dec_score_head = nn.ModuleList(
#             [nn.Linear(hidden_dim, nc) for _ in range(num_pred_heads)]
#         )
#         self.dec_bbox_head = nn.ModuleList(
#             [MLP(hidden_dim, hidden_dim, 4, num_layers=bbox_head_num_layers, act=nn.SiLU) for _ in range(num_pred_heads)]
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Adopt RT-DETR's initialization for matching components
#         bias_cls = bias_init_with_prob(0.01) # RT-DETR uses 0.01
#         # if self.nc != 80: # Ultralytics RT-DETR adjusts this if nc is not 80.
#         #    bias_cls = bias_cls / 80 * self.nc # Optional adjustment

#         # Encoder head
#         if hasattr(self.enc_output[0], 'weight'): # Linear layer
#             xavier_uniform_(self.enc_output[0].weight)
#         # constant_(self.enc_score_head.bias, bias_cls) # Done in loop below for all score heads
#         if hasattr(self.enc_bbox_head.layers[-1], 'weight'): # Last layer of MLP
#             constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
#             constant_(self.enc_bbox_head.layers[-1].bias, 0.0)

#         # Decoder/Query Embeddings
#         if self.learnt_init_query:
#             xavier_uniform_(self.tgt_embed.weight)
#         if hasattr(self.query_pos_head.layers[0], 'weight'): # First layer of MLP
#             xavier_uniform_(self.query_pos_head.layers[0].weight)
#         if len(self.query_pos_head.layers) > 1 and hasattr(self.query_pos_head.layers[1], 'weight'): # Second layer
#              xavier_uniform_(self.query_pos_head.layers[1].weight)


#         # Final prediction heads (dec_score_head, dec_bbox_head)
#         for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
#             constant_(cls_.bias, bias_cls)
#             if hasattr(reg_.layers[-1], 'weight'): # Last layer of MLP
#                 constant_(reg_.layers[-1].weight, 0.0)
#                 constant_(reg_.layers[-1].bias, 0.0)
        
#         # Also initialize enc_score_head bias (if not covered by dec_score_head loop, e.g. if not return_intermediate)
#         constant_(self.enc_score_head.bias, bias_cls)


#         # Initialize other ViL-specific layers (example, adjust as needed)
#         # Your original code initializes all nn.Linear with xavier_uniform_.
#         # This might be okay for the ViL blocks if they don't have specific needs.
#         # Or, ViLLayerFusedQKV and RefactoredxLSTMCrossReader should have their own reset_parameters method.
#         for m in self.writer.modules():
#             if hasattr(m, 'reset_parameters') and m != self: m.reset_parameters()
#         for m in self.reader.modules():
#             if hasattr(m, 'reset_parameters') and m != self: m.reset_parameters()
#         if hasattr(self.fuse_to_single, 'weight'):
#              xavier_uniform_(self.fuse_to_single.weight)
#              if self.fuse_to_single.bias is not None:
#                  constant_(self.fuse_to_single.bias, 0.0)


#     def _generate_anchors(self, B, dtype, device, eps=1e-2): # Matched signature from RT-DETR
#         # Generates anchors for the single fused scale, but applies logit transformation
#         # Matching RT-DETR's _generate_anchors structure, adapted for single scale
#         shapes = self.vil_fpn_spatial_shapes.to(device) # Should be [[fuse_out_size, fuse_out_size]]
#         anchors = []
#         # i will only be 0
#         for i, (H_fused, W_fused) in enumerate(shapes):
#             # Meshgrid (gx, gy)
#             sy = torch.arange(end=H_fused.item(), dtype=dtype, device=device)
#             sx = torch.arange(end=W_fused.item(), dtype=dtype, device=device)
#             # Consider torch.meshgrid indexing if needed for older PyTorch
#             grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
#             grid_xy = torch.stack([grid_x, grid_y], -1)  # [H_fused, W_fused, 2]

#             valid_WH = torch.tensor([W_fused, H_fused], dtype=dtype, device=device)
#             # Normalized cx, cy
#             grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH # [1, H_fused, W_fused, 2]
#             # Anchor wh
#             # In RT-DETR: grid_size * (2.0**i). Here i is always 0 for the single fused scale.
#             wh = torch.ones_like(grid_xy) * self.grid_size_anchor # [1, H_fused, W_fused, 2]
            
#             # Concatenate cxcywh and flatten
#             current_anchors = torch.cat([grid_xy, wh], -1).view(1, -1, 4) # [1, H_fused*W_fused, 4]
#             anchors.append(current_anchors)
        
#         anchors = torch.cat(anchors, 1) # [1, num_total_anchors, 4] (num_total_anchors = H_fused*W_fused)

#         # Apply valid_mask and logit transformation like in RT-DETR
#         valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
#         anchors = torch.log(anchors / (1 - anchors)) # To logit space
#         anchors = anchors.masked_fill(~valid_mask, float("inf"))
        
#         return anchors.repeat(B, 1, 1), valid_mask.repeat(B, 1, 1) # Repeat for batch

#     def _initial_queries(self, fused_map, batch): # Fused map is [B, hidden_dim, H_fused, W_fused]
#             B, C, H_fused, W_fused = fused_map.shape
            
#             # Flatten: [B, H_fused*W_fused, hidden_dim]
#             flattened_features = fused_map.flatten(2).permute(0, 2, 1)

#             # Generate anchors (now in logit space)
#             anchors_logit, valid_mask = self._generate_anchors(B, flattened_features.dtype, flattened_features.device)
            
#             # Apply encoder heads to flattened features
#             transformed_features = self.enc_output(valid_mask * flattened_features)

#             scores_logits = self.enc_score_head(transformed_features)
#             boxes_delta_logit = self.enc_bbox_head(transformed_features)
            
#             initial_boxes_logit = boxes_delta_logit + anchors_logit
#             initial_boxes_sigmoid = initial_boxes_logit.sigmoid()

#             ntok = flattened_features.shape[1]
#             actual_k = min(self.num_queries, ntok)
            
#             topk_scores, topk_indices = torch.topk(scores_logits.max(-1).values, actual_k, dim=1)

#             expanded_topk_indices_feat = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
#             sel_feat_content = torch.gather(transformed_features, 1, expanded_topk_indices_feat)
            
#             expanded_topk_indices_box = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
#             sel_box_sigmoid = torch.gather(initial_boxes_sigmoid, 1, expanded_topk_indices_box)
#             sel_box_logit = torch.gather(initial_boxes_logit, 1, expanded_topk_indices_box)

#             expanded_topk_indices_score = topk_indices.unsqueeze(-1).expand(-1, -1, self.nc)
#             sel_score_logits = torch.gather(scores_logits, 1, expanded_topk_indices_score)

#             # --- Denoising ---
#             dn_embed_content, dn_bbox_logit, attn_mask, dn_meta = get_cdn_group(
#                 batch, self.nc, actual_k,
#                 self.denoising_class_embed.weight if self.denoising_class_embed else None,
#                 self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training
#             )

#             # Prepare query features and reference boxes for ViL decoder
#             if self.learnt_init_query:
#                 query_content_features_base = self.tgt_embed.weight.unsqueeze(0).repeat(B, 1, 1)
#             else:
#                 query_content_features_base = sel_feat_content

#             reference_boxes_logit_base = sel_box_logit

#             # IMPORTANT: Detach if in training mode, and not learned queries
#             if self.training:
#                 if not self.learnt_init_query: # Only detach if not learnt
#                     query_content_features_base = query_content_features_base.detach()
#                 reference_boxes_logit_base = reference_boxes_logit_base.detach()
#                 if dn_embed_content is not None: # Only detach if denoising is actually present
#                     dn_embed_content = dn_embed_content.detach()
#                 if dn_bbox_logit is not None: # Only detach if denoising is actually present
#                     dn_bbox_logit = dn_bbox_logit.detach()


#             # Concatenate denoising queries for the decoder input
#             # Check if dn_embed_content and dn_bbox_logit are not None,
#             # which implies denoising is active and returned valid tensors.
#             if dn_embed_content is not None and dn_bbox_logit is not None:
#                 q_content_for_decoder = torch.cat([dn_embed_content, query_content_features_base], dim=1)
#                 q_box_logit_for_decoder = torch.cat([dn_bbox_logit, reference_boxes_logit_base], dim=1)
#             else:
#                 q_content_for_decoder = query_content_features_base
#                 q_box_logit_for_decoder = reference_boxes_logit_base
#                 # If no denoising, ensure dn_meta is also correctly indicating no denoising
#                 if dn_meta is not None: # dn_meta could be None if self.training is False
#                     dn_meta['num_denoising_queries'] = 0
#                     dn_meta['dn_num_split'][0] = 0
#                 # If dn_meta is None (e.g., self.training is False), no need to set anything

#             # --- Fix for KeyError: 'num_denoising_queries' ---
#             # Get the actual number of denoising queries from dn_meta.
#             # Only access dn_meta if it's not None and we are in training.
#             current_num_denoising_queries = 0
#             if self.training and dn_meta is not None and 'num_denoising_queries' in dn_meta:
#                 current_num_denoising_queries = dn_meta['num_denoising_queries']
#             # --- End Fix ---

#             if self.training and current_num_denoising_queries > 0:
#                 # Create `enc_boxes_for_loss` that includes denoising part, converted to sigmoid
#                 enc_boxes_for_loss = torch.cat([dn_bbox_logit.sigmoid(), sel_box_sigmoid], dim=1)

#                 placeholder_dn_scores = torch.full(
#                     (B, current_num_denoising_queries, self.nc),
#                     0.0, # Placeholder value
#                     device=sel_score_logits.device,
#                     dtype=sel_score_logits.dtype
#                 )
#                 enc_scores_for_loss = torch.cat([placeholder_dn_scores, sel_score_logits], dim=1)
#             else:
#                 enc_boxes_for_loss = sel_box_sigmoid
#                 enc_scores_for_loss = sel_score_logits

#             return q_content_for_decoder, q_box_logit_for_decoder, enc_boxes_for_loss, enc_scores_for_loss, dn_meta


#     def forward(self, x, batch=None):  # x is list of PAFPN outputs
#         # --- FUSION: PAFPN multi-scale -> single [B, hidden_dim, H_fused, W_fused] map ---
#         # This is your specific part
#         resized = [
#             F.adaptive_avg_pool2d(f, (self.fuse_out_size, self.fuse_out_size)) if f.shape[-1] > self.fuse_out_size
#             else F.interpolate(f, size=(self.fuse_out_size, self.fuse_out_size), mode='nearest') if f.shape[-1] < self.fuse_out_size
#             else f
#             for f_idx, f in enumerate(x)  # x is assumed to be a list of features from backbone
#         ]
#         fused_cat = torch.cat(resized, dim=1)
#         fused_map = self.fuse_to_single(fused_cat)  # [B, hidden_dim, H_fused, W_fused]
#         # -------------------------------------------

#         # ——— RMSNorm on fused feature map ———
#         # switch to channel-last for RMSNorm over channels
#         fused_map = fused_map.permute(0, 2, 3, 1)  # [B, H_fused, W_fused, hidden_dim]
#         fused_map = self.fuse_norm(fused_map)      # normalize over hidden_dim
#         fused_map = fused_map.permute(0, 3, 1, 2)  # back to [B, hidden_dim, H_fused, W_fused]
#         # ----------------------------------------------------------------

#         # Initial queries from the fused map (encoder proposals)
#         # q_content, q_box_logit, enc_boxes_sigmoid, enc_scores_logits, dn_meta
#         q_content, qb_logit, enc_boxes, enc_scores, dn_meta = self._initial_queries(fused_map, batch)
        
#         # Add positional embedding from boxes to content features
#         # qb_logit is [B, N_queries_total, 4]
#         # query_pos_head expects normalized boxes, but RT-DETR passes logit boxes and hopes MLP learns.
#         # Let's pass sigmoided boxes to query_pos_head if it expects normalized [0,1]
#         # Or, ensure query_pos_head can handle logit.
#         # Given RT-DETR structure, query_pos_head is applied to reference points (logit or sigmoid)
#         # If query_pos_head is like RT-DETR's, it expects 4D input.
#         # Let's assume it works fine on logit_boxes, or sigmoid(logit_boxes)
#         # Common practice is to use sigmoid(logit_boxes) for the positional embedding generation.
#         pos_embed = self.query_pos_head(qb_logit.sigmoid())  # Or qb_logit directly if MLP handles it
#         q_final_content = q_content + pos_embed               # These are the queries for ViL blocks

#         # ——— RMSNorm on decoder queries ———
#         q_final_content = self.decoder_norm(q_final_content)
#         # ----------------------------------------------------------------

#         # Memory for ViL blocks (from the single fused map)
#         memory = fused_map.permute(0, 2, 3, 1).reshape(fused_map.size(0), -1, self.hidden_dim)

#         # --- ViL Decoder Iterations (Your Specific Part) ---
#         # dn_len and det_len from dn_meta
#         dn_len = dn_meta['num_denoising_queries'] if dn_meta and 'num_denoising_queries' in dn_meta else 0
#         # det_len = dn_meta['num_matching_queries'] if dn_meta else q_final_content.size(1) - dn_len
        
#         # q_final_content is now [B, total_queries, C]
#         # Split into dn_queries and det_queries for ViL processing if ViL needs it
#         current_queries = q_final_content  # This is `tgt` or `embed` in RT-DETR decoder

#         all_layer_outputs_scores_logits = []
#         all_layer_outputs_boxes_sigmoid = []

#         # Loop through ViL layers
#         for i in range(self.num_vil_layers):
#             if dn_len > 0:
#                 dn_queries = current_queries[:, :dn_len, :]
#                 det_queries = current_queries[:, dn_len:, :]
#                 dn_queries_self_attended = self.writer[i](dn_queries)
#                 det_queries_self_attended = self.writer[i](det_queries)
#                 queries_for_cross_attention = torch.cat([dn_queries_self_attended, det_queries_self_attended], dim=1)
#             else:
#                 queries_for_cross_attention = self.writer[i](current_queries)
#             current_queries = self.reader[i](memory, queries_for_cross_attention)
            
#             head_idx = i if self.return_intermediate_outputs else 0
#             layer_scores_logits = self.dec_score_head[head_idx](current_queries)
#             layer_boxes_delta_logit = self.dec_bbox_head[head_idx](current_queries)
#             layer_boxes_logit = qb_logit + layer_boxes_delta_logit
#             layer_boxes_sigmoid = layer_boxes_logit.sigmoid()
            
#             all_layer_outputs_scores_logits.append(layer_scores_logits)
#             all_layer_outputs_boxes_sigmoid.append(layer_boxes_sigmoid)
            
#             if self.training:
#                 qb_logit = layer_boxes_logit.detach()
#             else:
#                 qb_logit = layer_boxes_logit
        
#         # Stack intermediate outputs (matching RT-DETR output format for decoder)
#         # [num_layers, B, N_queries_total, C]
#         dec_scores_logits = torch.stack(all_layer_outputs_scores_logits)
#         dec_boxes_sigmoid = torch.stack(all_layer_outputs_boxes_sigmoid)

#         # Prepare final output (matching RT-DETR)
#         # enc_boxes and enc_scores are from the initial_queries stage
#         # dec_boxes_sigmoid and dec_scores_logits are from the ViL decoder stages
        
#         # k_enc = enc_scores.size(1) # Number of proposals from encoder stage
#         # For loss calculation, enc_scores and enc_boxes are needed.
#         # RT-DETR might select top-k from enc_scores/enc_boxes for loss if different from num_decoder_queries
#         # Here, enc_scores/enc_boxes from _initial_queries are already top-k based on num_queries
        
#         if self.training:
#             return dec_boxes_sigmoid, dec_scores_logits, enc_boxes, enc_scores, dn_meta
#         else:
#             # Inference: use outputs from the last ViL layer
#             last_dec_boxes_sigmoid = dec_boxes_sigmoid[-1]  # [B, N_queries_total, 4]
#             last_dec_scores_logits = dec_scores_logits[-1]  # [B, N_queries_total, nc]

#             # Remove denoising queries for final output if they exist
#             if dn_meta and dn_meta.get('num_denoising_queries', 0) > 0:
#                 dn_queries_count = dn_meta['num_denoising_queries']
#                 # Output only detection queries
#                 final_boxes = last_dec_boxes_sigmoid[:, dn_queries_count:, :]
#                 final_scores = last_dec_scores_logits[:, dn_queries_count:, :].sigmoid()  # Apply sigmoid for scores
#             else:
#                 final_boxes = last_dec_boxes_sigmoid
#                 final_scores = last_dec_scores_logits.sigmoid()

#             # Concatenate boxes and scores: [B, N_det_queries, 4+nc]
#             # This matches Ultralytics RT-DETR's inference output format
#             # Note: RT-DETR output for export might be squeezed if B=1
#             # y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
#             # Here, final_boxes/final_scores are already [B, N, Dim]
#             output = torch.cat([final_boxes, final_scores], dim=-1)
            
#             # Original RT-DETR returns (y, (all_outputs_for_eval_if_needed))
#             # For simplicity, just return concatenated output, or adapt as needed.
#             all_outputs_tuple = (dec_boxes_sigmoid, dec_scores_logits, enc_boxes, enc_scores, dn_meta)
#             return output, all_outputs_tuple  # Or just output
        

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

# -----------------------------------------------------------------------------
# Project‑local imports (must exist in your repo)
# -----------------------------------------------------------------------------
#   • ViLLayerFusedQKV, SequenceTraversal
#   • FeedForward, MatrixLSTMCell
#   • SequenceConv2d, MLP, bias_init_with_prob, get_cdn_group
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2‑D sine/cos positional embedding (fixed, non‑trainable)
# -----------------------------------------------------------------------------

def _build_2d_sincos_embedding(h: int, w: int, dim: int, *, temperature: float = 10000.0) -> torch.Tensor:
    """Return (1, dim, h, w) tensor with classic 2‑D sine‑cos pose."""
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for 2‑D PE")
    y = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
    x = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
    dim_t = temperature ** (torch.arange(dim // 2) / (dim // 2))
    pos_x = (x.unsqueeze(-1) / dim_t).reshape(h, w, -1)
    pos_y = (y.unsqueeze(-1) / dim_t).reshape(h, w, -1)
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), -1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), -1).flatten(-2)
    return torch.cat((pos_y, pos_x), -1).permute(2, 0, 1).unsqueeze(0)

# =============================================================================
#                      ViL‑RT‑DETR Head (multi‑scale memory)
# =============================================================================

from .vision_lstm.vision_lstm2 import *
# 3rd-party helpers already in your repo ------------------------------------
# * ViLLayerFusedQKV, RefactoredxLSTMCrossReader, SequenceTraversal
# * MLP, bias_init_with_prob, get_cdn_group
# -------------------------------------------------------------------------- #

def _build_2d_sincos_embedding(dim: int,
                               h:   int,
                               w:   int,
                               temperature: float = 10000.0) -> torch.Tensor:
    """
    Return a **(C,H,W)** fixed sin-cos positional table.
    Channel layout is identical to DETR / ViT-Det.
    """
    assert dim % 4 == 0, "`dim` must be a multiple of 4"
    quarter = dim // 4
    freq    = temperature ** (-torch.arange(quarter, dtype=torch.float32) / quarter)  # (C/4,)
    gy, gx  = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')          # (H,W)

    pos_x   = gx[:, :, None] * freq                               # (H,W,C/4)
    pos_x   = torch.stack((pos_x.sin(), pos_x.cos()), -1).flatten(-2)  # (H,W,C/2)
    pos_y   = gy[:, :, None] * freq
    pos_y   = torch.stack((pos_y.sin(), pos_y.cos()), -1).flatten(-2)

    return torch.cat((pos_y, pos_x), -1).permute(2, 0, 1)          # (C,H,W)


# -------------------------------------------------------------------------- #
#   generate (memory_direction, query_direction) pairs for n decoder layers #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# 1. fix traversal-pair generator
def _reader_traversal_pairs(n_layers: int,
                            Traversal,
                            mem_flip_every: int = 1,
                            qry_flip_every: int = 2):
    pairs   = []
    mem_dir = qry_dir = Traversal.ROWWISE_FROM_TOP_LEFT
    for i in range(n_layers):
        if i > 0 and (i % mem_flip_every == 0):
            mem_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if mem_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        if i > 0 and (i % qry_flip_every == 0):          #  <-- add  i > 0
            qry_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if qry_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        pairs.append((mem_dir, qry_dir))
    return pairs


from math import sqrt
# -------------------------------------------------------------------------- #
#                           ViL-based RT-DETR head                           #
# -------------------------------------------------------------------------- #
class ViLBasedRTDETRHead(nn.Module):
    """
    RT-DETR detection head powered by ViL blocks.
    • four independent FPN levels → concatenated memory sequence
    • no extra FPN fusion - each scale is processed in-place
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 ch_vil,
                 memory_spatial_shapes,
                 hidden_dim                  = 256,
                 num_queries                 = 300,
                 num_vil_layers              = 6,
                 vil_block_decoder_args      = None,
                 act                         = nn.SiLU,
                 nc                          = 80,
                 # denoising ------------------------------------------------
                 num_denoising               = 100,
                 label_noise_ratio           = 0.5,
                 box_noise_scale             = 1.0,
                 # anchors --------------------------------------------------
                 grid_size_anchor            = 0.05,
                 # queries --------------------------------------------------
                 learnt_init_query           = False,
                 query_pos_head_num_layers   = 2,
                 bbox_head_num_layers        = 3,
                 return_intermediate_outputs = True):
        super().__init__()

        # ---------------- bookkeeping ---------------------------------- #
        self.memory_spatial_shapes = [tuple(map(int, s)) for s in memory_spatial_shapes]
        assert len(ch_vil) == len(self.memory_spatial_shapes)
        self.num_scales   = len(self.memory_spatial_shapes)

        self.hidden_dim   = int(hidden_dim)
        self.num_queries  = int(num_queries)
        self.num_vil_layers = int(num_vil_layers)
        self.return_intermediate_outputs = bool(return_intermediate_outputs)
        self.nc           = int(nc)
        self.grid_size_anchor = float(grid_size_anchor)

        # ---------------- per-level linear projection ------------------ #
        self.scale_proj = nn.ModuleList(nn.Conv2d(c, self.hidden_dim, 1) for c in ch_vil)
        self.scale_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=True)

        # ---------------- fixed PE + level embedding ------------------- #
        self.pos_embed_tables = nn.ParameterList([
            nn.Parameter(_build_2d_sincos_embedding(self.hidden_dim, h, w),
                         requires_grad=False)
            for (h, w) in self.memory_spatial_shapes
        ])
        self.level_embed = nn.Parameter(torch.zeros(self.num_scales, self.hidden_dim))

        self.register_buffer('vil_fpn_spatial_shapes',
                             torch.tensor(self.memory_spatial_shapes, dtype=torch.long),
                             persistent=False)

        # ---------------- encoder proposal heads ----------------------- #
        self.enc_output     = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.LayerNorm(self.hidden_dim))
        self.enc_score_head = nn.Linear(self.hidden_dim, self.nc)
        self.enc_bbox_head  = MLP(self.hidden_dim, self.hidden_dim, 4,
                                  num_layers=bbox_head_num_layers, act=nn.SiLU)

        # ---------------- denoising ------------------------------------ #
        self.num_denoising     = int(num_denoising)
        self.label_noise_ratio = float(label_noise_ratio)
        self.box_noise_scale   = float(box_noise_scale)
        self.denoising_class_embed = (
            nn.Embedding(self.nc, self.hidden_dim) if self.num_denoising else None
        )

        # ---------------- query embeddings ----------------------------- #
        self.learnt_init_query = bool(learnt_init_query)
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        self.query_pos_head = MLP(4, 2 * self.hidden_dim, self.hidden_dim,
                                  num_layers=query_pos_head_num_layers, act=nn.SiLU)
        self.decoder_norm   = nn.LayerNorm(self.hidden_dim, eps=1e-6, elementwise_affine=True)

        # ---------------- ViL writer / reader stacks ------------------- #
        base_cfg = dict(dim=self.hidden_dim,
                        expansion=2,
                        qkv_block_size=max(1, self.hidden_dim // 64))
        if vil_block_decoder_args:
            base_cfg.update(vil_block_decoder_args)


        # writer (self-attention in query space) – alternate direction each layer
        self.writer = nn.ModuleList(
            ViLLayerFusedQKV(**base_cfg,
                             direction=(SequenceTraversal.ROWWISE_FROM_TOP_LEFT
                                        if i % 2 == 0
                                        else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT))
            for i in range(self.num_vil_layers)
        )

        # reader (cross-attend Q ← memory) – visit **all 4 traversal combos**
        dir_pairs = _reader_traversal_pairs(self.num_vil_layers,
                                            SequenceTraversal,
                                            mem_flip_every=1,
                                            qry_flip_every=2)
        self.reader = nn.ModuleList(
            RefactoredxLSTMCrossReader(dim=self.hidden_dim,
                                       expansion=base_cfg['expansion'],
                                       qkv_block_size=base_cfg['qkv_block_size'],
                                       seqlens_list=self.memory_spatial_shapes,
                                       memory_direction=m_dir,
                                       query_direction=q_dir)
            for m_dir, q_dir in dir_pairs
        )

        # ---------------- per-layer prediction heads ------------------- #
        heads = self.num_vil_layers if self.return_intermediate_outputs else 1
        self.dec_score_head = nn.ModuleList(nn.Linear(self.hidden_dim, self.nc)
                                            for _ in range(heads))
        self.dec_bbox_head  = nn.ModuleList(MLP(self.hidden_dim, self.hidden_dim, 4,
                                                num_layers=bbox_head_num_layers,
                                                act=nn.SiLU)
                                            for _ in range(heads))

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    def _reset_parameters(self):
        #bias_cls = bias_init_with_prob(0.01)
        bias_cls = bias_init_with_prob(0.01) * sqrt(self.hidden_dim / 256)
        if hasattr(self.enc_output[0], 'weight'):
            xavier_uniform_(self.enc_output[0].weight)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias,   0.0)
        constant_(self.enc_score_head.bias, bias_cls)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        if hasattr(self.query_pos_head.layers[0], 'weight'):
            xavier_uniform_(self.query_pos_head.layers[0].weight)
        for cls_h, reg_h in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_h.bias, bias_cls)
            constant_(reg_h.layers[-1].weight, 0.0)
            constant_(reg_h.layers[-1].bias,   0.0)
        nn.init.normal_(self.level_embed, std=0.02)
        for proj in self.scale_proj:
            xavier_uniform_(proj.weight)
            if proj.bias is not None:
                constant_(proj.bias, 0.0)

    # ------------------------------------------------------------------ #
    #                           anchor helper                            #
    # ------------------------------------------------------------------ #
    def _generate_anchors(self, shapes, B, dtype, device, eps=1e-2):
        anchors, valid = [], []
        for lvl, (H, W) in enumerate(shapes):
            gy, gx = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                                    torch.arange(W, dtype=dtype, device=device),
                                    indexing='ij')
            grid   = torch.stack([gx, gy], -1).unsqueeze(0)                # (1,H,W,2)
            grid   = (grid + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
            wh     = torch.ones_like(grid) * self.grid_size_anchor * (2.0 ** lvl)
            a      = torch.cat([grid, wh], -1).view(1, -1, 4)
            anchors.append(a)
            valid.append(((a > eps) & (a < 1 - eps)).all(-1, keepdim=True))
        anchors = torch.cat(anchors, 1)           # (1,N,4)
        valid   = torch.cat(valid,   1)           # (1,N,1)
        anchors = torch.log(anchors / (1 - anchors)).masked_fill(~valid, float('inf'))
        return anchors.repeat(B, 1, 1), valid.repeat(B, 1, 1)

    # ------------------------------------------------------------------ #
    #                       encoder → top-K queries                      #
    # ------------------------------------------------------------------ #
    def _initial_queries(self, feats, shapes, batch):
        B, N, _ = feats.shape
        anchors, valid = self._generate_anchors(shapes, B, feats.dtype, feats.device)

        enc_feats  = self.enc_output(valid * feats)       # (B,N,C)
        cls_logits = self.enc_score_head(enc_feats)       # (B,N,nc)
        box_logit  = self.enc_bbox_head(enc_feats) + anchors

        k = min(self.num_queries, N)
        topk = torch.topk(cls_logits.max(-1).values, k, dim=1).indices
        gather = lambda src, d: torch.gather(src, 1,
                                             topk.unsqueeze(-1).expand(-1, -1, d))
        sel_feat   = gather(enc_feats, self.hidden_dim)
        sel_box    = gather(box_logit, 4)
        sel_score  = gather(cls_logits, self.nc)

        # ------------ denoising --------------------------------------- #
        dn_c, dn_b, _, dn_meta = get_cdn_group(
            batch, self.nc, k,
            self.denoising_class_embed.weight if self.denoising_class_embed else None,
            self.num_denoising, self.label_noise_ratio,
            self.box_noise_scale, self.training)

        base_q = self.tgt_embed.weight.unsqueeze(0).repeat(B,1,1) if self.learnt_init_query else sel_feat
        base_b = sel_box
        if self.training:
            if not self.learnt_init_query:
                base_q = base_q
            base_b = base_b

        if dn_c is not None and dn_b is not None:
            q_content = torch.cat([dn_c, base_q], 1)
            q_boxes   = torch.cat([dn_b, base_b], 1)
        else:
            q_content, q_boxes = base_q, base_b
            if dn_meta:
                dn_meta['num_denoising_queries'] = 0
                dn_meta['dn_num_split'][0]       = 0

        # encoder outputs for loss ------------------------------------- #
        dn_len = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        if self.training and dn_len > 0:
            zeros      = torch.zeros(B, dn_len, self.nc,
                                      device=cls_logits.device, dtype=cls_logits.dtype)
            enc_scores = torch.cat([zeros, sel_score], 1)
            enc_boxes  = torch.cat([dn_b.sigmoid(), sel_box.sigmoid()], 1)
        else:
            enc_scores, enc_boxes = sel_score, sel_box.sigmoid()

        return q_content, q_boxes, enc_boxes, enc_scores, dn_meta

    # ------------------------------------------------------------------ #
    #                            forward pass                            #
    # ------------------------------------------------------------------ #
    def forward(self, feats_per_level, batch=None):
        """
        *feats_per_level* – list of 4 tensors from ViL-FPN
            [(B,C,H,W), ...] ordered high->low resolution.
        """
        B = feats_per_level[0].size(0)

        # ------------ build concatenated memory ----------------------- #
        mem_parts = []
        for lvl, feat in enumerate(feats_per_level):
            feat = self.scale_proj[lvl](feat)
            feat = feat + self.pos_embed_tables[lvl].unsqueeze(0).to(feat.dtype)
            feat = feat + self.level_embed[lvl].view(1, -1, 1, 1)
            feat = self.scale_norm(feat.permute(0,2,3,1)).permute(0,3,1,2)
            mem_parts.append(feat.permute(0,2,3,1).reshape(B, -1, self.hidden_dim))
        memory = torch.cat(mem_parts, 1)                             # (B,N,C)

        q_cnt, q_box, enc_boxes, enc_scores, dn_meta = \
            self._initial_queries(memory, self.memory_spatial_shapes, batch)

        queries = self.decoder_norm(q_cnt + self.query_pos_head(q_box.sigmoid()))
        dn_len  = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0

        dec_scores, dec_boxes = [], []
        cur_q, cur_box = queries, q_box
        for i in range(self.num_vil_layers):
            # writer (self-attention)
            if dn_len:
                cur_q = torch.cat([self.writer[i](cur_q[:, :dn_len]),
                                   self.writer[i](cur_q[:, dn_len:])], 1)
            else:
                cur_q = self.writer[i](cur_q)

            # reader (cross-attention)
            cur_q = self.reader[i](memory, cur_q)

            head_idx = i if self.return_intermediate_outputs else 0
            cls_log   = self.dec_score_head[head_idx](cur_q)
            cur_box   = cur_box + self.dec_bbox_head[head_idx](cur_q)

            dec_scores.append(cls_log)
            dec_boxes .append(cur_box.sigmoid())
            # if self.training:
            #     cur_box = cur_box #.detach()

        dec_scores = torch.stack(dec_scores)          # (L,B,N,C)
        dec_boxes  = torch.stack(dec_boxes)           # (L,B,N,4)

        # ---------------- output -------------------------------------- #
        if self.training:
            return dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta

        final_scores, final_boxes = dec_scores[-1], dec_boxes[-1]
        if dn_len:
            final_scores, final_boxes = final_scores[:, dn_len:], final_boxes[:, dn_len:]

        det_out = torch.cat([final_boxes, final_scores.sigmoid()], -1)  # (B,N,4+nc)
        return det_out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)



# -------------------------------------------------------------------------- #
# 2-D sine/cos positional embedding (fixed, non-trainable)
# -------------------------------------------------------------------------- #
def _build_2d_sincos_embedding(dim: int,
                               h:   int,
                               w:   int,
                               temperature: float = 10000.0) -> torch.Tensor:
    assert dim % 4 == 0, "`dim` must be a multiple of 4"
    quarter = dim // 4
    freq    = temperature ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
    gy, gx  = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    pos_x   = gx[:, :, None] * freq
    pos_x   = torch.stack((pos_x.sin(), pos_x.cos()), -1).flatten(-2)
    pos_y   = gy[:, :, None] * freq
    pos_y   = torch.stack((pos_y.sin(), pos_y.cos()), -1).flatten(-2)

    return torch.cat((pos_y, pos_x), -1).permute(2, 0, 1)  # (C,H,W)

# -------------------------------------------------------------------------- #
# generate (memory_direction, query_direction) pairs for decoder layers
# -------------------------------------------------------------------------- #
def _reader_traversal_pairs(n_layers: int,
                            Traversal,
                            mem_flip_every: int = 1,
                            qry_flip_every: int = 2):
    pairs   = []
    mem_dir = qry_dir = Traversal.ROWWISE_FROM_TOP_LEFT
    for i in range(n_layers):
        if i > 0 and (i % mem_flip_every == 0):
            mem_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if mem_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        if i > 0 and (i % qry_flip_every == 0):
            qry_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if qry_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        pairs.append((mem_dir, qry_dir))
    return pairs

# -------------------------------------------------------------------------- #
# Single-Scale Fusion Module
# -------------------------------------------------------------------------- #
class SingleScaleFusion(nn.Module):
    """
    Fuse multiple FPN feature maps into one single-scale feature map.

    Args:
        in_channels_list: List[int] of channel dims for each input feature.
        out_dim:          int, number of output channels (d_model).
        target_size:      tuple (H, W) the resolution to which all inputs are resized.
    """
    def __init__(self,
                 in_channels_list: list[int],
                 out_dim: int,
                 target_size: tuple[int, int]):
        super().__init__()
        self.target_size = target_size  # (H, W)
        total_in = sum(in_channels_list)
        self.fuse_conv = nn.Conv2d(total_in, out_dim, kernel_size=1)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        H, W = self.target_size
        resized = [
            F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            for f in features
        ]
        x = torch.cat(resized, dim=1)    # [B, sum(C_i), H, W]
        return self.fuse_conv(x)         # [B, out_dim, H, W]

# -------------------------------------------------------------------------- #
# ViL-based RT-DETR head with optional single-scale fusion
# -------------------------------------------------------------------------- #
class ViLBasedRTDETRHead(nn.Module):
    """
    RT-DETR detection head powered by ViL blocks.
    Can operate on multi-scale FPN outputs or a single fused scale.
    """
    def __init__(self,
                 ch_vil: list[int],
                 memory_spatial_shapes: list[list[int,int]],
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 num_vil_layers: int = 6,
                 vil_block_decoder_args: dict = None,
                 act=nn.SiLU,
                 nc: int = 80,
                 use_single_scale: bool = False,
                 fusion_target_size: list[int,int] = [20,20],
                 num_denoising: int = 100,
                 label_noise_ratio: float = 0.5,
                 box_noise_scale: float = 1.0,
                 grid_size_anchor: float = 0.05,
                 learnt_init_query: bool = False,
                 query_pos_head_num_layers: int = 2,
                 bbox_head_num_layers: int = 3,
                 return_intermediate_outputs: bool = True):
        super().__init__()
        self.use_single_scale = use_single_scale


        # fusion config
        if self.use_single_scale:
            self.fuser = SingleScaleFusion(
                in_channels_list=ch_vil,
                out_dim=hidden_dim,
                target_size=fusion_target_size
            )
            ch_vil = [hidden_dim]
            self.memory_spatial_shapes = [fusion_target_size]

        self.hidden_dim   = hidden_dim
        self.num_queries  = num_queries
        self.num_vil_layers = num_vil_layers
        self.return_intermediate_outputs = return_intermediate_outputs
        self.nc           = nc
        self.grid_size_anchor = grid_size_anchor

        # per-level projection
        self.scale_proj = nn.ModuleList(nn.Conv2d(c, hidden_dim, 1) for c in ch_vil)
        self.scale_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)

        # positional + level embeddings
        self.pos_embed_tables = nn.ParameterList([
            nn.Parameter(_build_2d_sincos_embedding(hidden_dim, h, w), requires_grad=False)
            for (h, w) in self.memory_spatial_shapes
        ])
        self.level_embed = nn.Parameter(torch.zeros(len(self.memory_spatial_shapes), hidden_dim))
        self.register_buffer('vil_fpn_spatial_shapes',
                             torch.tensor(self.memory_spatial_shapes, dtype=torch.long),
                             persistent=False)

        # encoder proposal heads
        self.enc_output     = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head  = MLP(hidden_dim, hidden_dim, 4,
                                  num_layers=bbox_head_num_layers, act=nn.SiLU)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale   = box_noise_scale
        self.denoising_class_embed = (
            nn.Embedding(nc, hidden_dim) if num_denoising else None
        )

        # query embeddings
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2*hidden_dim, hidden_dim,
                                  num_layers=query_pos_head_num_layers, act=nn.SiLU)
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)

        # writer/reader
        base_cfg = dict(dim=hidden_dim, expansion=2, qkv_block_size=max(1, hidden_dim//64))
        if vil_block_decoder_args:
            base_cfg.update(vil_block_decoder_args)
        self.writer = nn.ModuleList(
            ViLLayerFusedQKV(**base_cfg,
                             direction=(SequenceTraversal.ROWWISE_FROM_TOP_LEFT
                                        if i%2==0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT))
            for i in range(num_vil_layers)
        )
        dir_pairs = _reader_traversal_pairs(num_vil_layers, SequenceTraversal, mem_flip_every=1, qry_flip_every=2)
        self.reader = nn.ModuleList(
            RefactoredxLSTMCrossReader(dim=hidden_dim,
                                       expansion=base_cfg['expansion'],
                                       qkv_block_size=base_cfg['qkv_block_size'],
                                       seqlens_list=self.memory_spatial_shapes,
                                       memory_direction=m_dir,
                                       query_direction=q_dir)
            for m_dir, q_dir in dir_pairs
        )

        # prediction heads
        heads = num_vil_layers if return_intermediate_outputs else 1
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(heads))
        self.dec_bbox_head  = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4,
                num_layers=bbox_head_num_layers, act=nn.SiLU)
            for _ in range(heads)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) * sqrt(self.hidden_dim/256)
        xavier_uniform_(self.enc_output[0].weight)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias,   0.0)
        constant_(self.enc_score_head.bias, bias_cls)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        if hasattr(self.query_pos_head.layers[0], 'weight'):
            xavier_uniform_(self.query_pos_head.layers[0].weight)
        for cls_h, reg_h in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_h.bias, bias_cls)
            constant_(reg_h.layers[-1].weight, 0.0)
            constant_(reg_h.layers[-1].bias,   0.0)
        nn.init.normal_(self.level_embed, std=0.02)
        for proj in self.scale_proj:
            xavier_uniform_(proj.weight)
            if proj.bias is not None:
                constant_(proj.bias, 0.0)

    def _generate_anchors(self, shapes, B, dtype, device, eps=1e-2):
        anchors, valid = [], []
        for lvl, (H, W) in enumerate(shapes):
            gy, gx = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                                    torch.arange(W, dtype=dtype, device=device),
                                    indexing='ij')
            grid = torch.stack([gx, gy], -1).unsqueeze(0)
            grid = (grid + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
            wh   = torch.ones_like(grid) * self.grid_size_anchor * (2.0 ** lvl)
            a    = torch.cat([grid, wh], -1).view(1, -1, 4)
            anchors.append(a)
            valid.append(((a > eps) & (a < 1 - eps)).all(-1, keepdim=True))
        anchors = torch.cat(anchors, 1)
        valid   = torch.cat(valid,   1)
        anchors = torch.log(anchors / (1 - anchors)).masked_fill(~valid, float('inf'))
        return anchors.repeat(B, 1, 1), valid.repeat(B, 1, 1)

    def _initial_queries(self, feats, shapes, batch):
        B, N, _ = feats.shape
        anchors, valid = self._generate_anchors(shapes, B, feats.dtype, feats.device)
        enc_feats  = self.enc_output(valid * feats)
        cls_logits = self.enc_score_head(enc_feats)
        box_logit  = self.enc_bbox_head(enc_feats) + anchors
        k = min(self.num_queries, N)
        topk = torch.topk(cls_logits.max(-1).values, k, dim=1).indices
        gather = lambda src, d: torch.gather(src, 1,
                                             topk.unsqueeze(-1).expand(-1, -1, d))
        sel_feat  = gather(enc_feats, self.hidden_dim)
        sel_box   = gather(box_logit, 4)
        sel_score = gather(cls_logits, self.nc)
        dn_c, dn_b, _, dn_meta = get_cdn_group(batch, self.nc, k,
                                                self.denoising_class_embed.weight if self.denoising_class_embed else None,
                                                self.num_denoising, self.label_noise_ratio,
                                                self.box_noise_scale, self.training)
        base_q = self.tgt_embed.weight.unsqueeze(0).repeat(B,1,1) if self.learnt_init_query else sel_feat
        base_b = sel_box
        if self.training:
            if not self.learnt_init_query:
                base_q = base_q
            base_b = base_b
        if dn_c is not None and dn_b is not None:
            q_content = torch.cat([dn_c, base_q], 1)
            q_boxes   = torch.cat([dn_b, base_b], 1)
        else:
            q_content, q_boxes = base_q, base_b
            if dn_meta:
                dn_meta['num_denoising_queries'] = 0
                dn_meta['dn_num_split'][0]       = 0
        dn_len = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        if self.training and dn_len > 0:
            zeros     = torch.zeros(B, dn_len, self.nc, device=cls_logits.device, dtype=cls_logits.dtype)
            enc_scores= torch.cat([zeros, sel_score], 1)
            enc_boxes = torch.cat([dn_b.sigmoid(), sel_box.sigmoid()], 1)
        else:
            enc_scores, enc_boxes = sel_score, sel_box.sigmoid()
        return q_content, q_boxes, enc_boxes, enc_scores, dn_meta

    def forward(self, feats_per_level, batch=None):
        B = feats_per_level[0].size(0)
        if self.use_single_scale:
            fused = self.fuser(*feats_per_level)
            x = self.scale_proj[0](fused)
            x = x + self.pos_embed_tables[0].unsqueeze(0).to(x.dtype)
            x = x + self.level_embed[0].view(1, -1, 1, 1)
            x = self.scale_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
            memory = x.reshape(B, -1, self.hidden_dim)
        else:
            mem_parts = []
            for lvl, feat in enumerate(feats_per_level):
                x = self.scale_proj[lvl](feat)
                x = x + self.pos_embed_tables[lvl].unsqueeze(0).to(x.dtype)
                x = x + self.level_embed[lvl].view(1, -1, 1, 1)
                x = self.scale_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
                mem_parts.append(x.reshape(B, -1, self.hidden_dim))
            memory = torch.cat(mem_parts, 1)
        q_cnt, q_box, enc_boxes, enc_scores, dn_meta = \
            self._initial_queries(memory, self.memory_spatial_shapes, batch)
        queries = self.decoder_norm(q_cnt + self.query_pos_head(q_box.sigmoid()))
        dn_len  = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        dec_scores, dec_boxes = [], []
        cur_q, cur_box = queries, q_box
        for i in range(self.num_vil_layers):
            if dn_len:
                cur_q = torch.cat([self.writer[i](cur_q[:, :dn_len]),
                                   self.writer[i](cur_q[:, dn_len:])], 1)
            else:
                cur_q = self.writer[i](cur_q)
            cur_q = self.reader[i](memory, cur_q)
            head_idx = i if self.return_intermediate_outputs else 0
            cls_log   = self.dec_score_head[head_idx](cur_q)
            cur_box   = cur_box + self.dec_bbox_head[head_idx](cur_q)
            dec_scores.append(cls_log)
            dec_boxes .append(cur_box.sigmoid())
        dec_scores = torch.stack(dec_scores)
        dec_boxes  = torch.stack(dec_boxes)
        if self.training:
            return dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta
        final_scores, final_boxes = dec_scores[-1], dec_boxes[-1]
        if dn_len:
            final_scores, final_boxes = final_scores[:, dn_len:], final_boxes[:, dn_len:]
        det_out = torch.cat([final_boxes, final_scores.sigmoid()], -1)
        return det_out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)

# -------------------------------------------------------------------------- #
# Note: ViLBasedRTDETRHead is agnostic to scale count; fusion flag toggles.   #
# -------------------------------------------------------------------------- #



# -------------------------------------------------------------------------- #
# 2-D sine/cos positional embedding (fixed, non-trainable)
# -------------------------------------------------------------------------- #
def _build_2d_sincos_embedding(dim: int,
                               h:   int,
                               w:   int,
                               temperature: float = 10000.0) -> torch.Tensor:
    assert dim % 4 == 0, "`dim` must be a multiple of 4"
    quarter = dim // 4
    freq    = temperature ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
    gy, gx  = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    pos_x   = gx[:, :, None] * freq
    pos_x   = torch.stack((pos_x.sin(), pos_x.cos()), -1).flatten(-2)
    pos_y   = gy[:, :, None] * freq
    pos_y   = torch.stack((pos_y.sin(), pos_y.cos()), -1).flatten(-2)

    return torch.cat((pos_y, pos_x), -1).permute(2, 0, 1)  # (C,H,W)

# -------------------------------------------------------------------------- #
# generate (memory_direction, query_direction) pairs for decoder layers
# -------------------------------------------------------------------------- #
def _reader_traversal_pairs(n_layers: int,
                            Traversal,
                            mem_flip_every: int = 1,
                            qry_flip_every: int = 2):
    pairs   = []
    mem_dir = qry_dir = Traversal.ROWWISE_FROM_TOP_LEFT
    for i in range(n_layers):
        if i > 0 and (i % mem_flip_every == 0):
            mem_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if mem_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        if i > 0 and (i % qry_flip_every == 0):
            qry_dir = (Traversal.ROWWISE_FROM_BOT_RIGHT
                       if qry_dir == Traversal.ROWWISE_FROM_TOP_LEFT
                       else Traversal.ROWWISE_FROM_TOP_LEFT)
        pairs.append((mem_dir, qry_dir))
    return pairs

# -------------------------------------------------------------------------- #
# Single-Scale Fusion Module
# -------------------------------------------------------------------------- #
class SingleScaleFusion(nn.Module):
    """
    Fuse multiple FPN feature maps into one single-scale feature map.

    Args:
        in_channels_list: List[int] of channel dims for each input feature.
        out_dim:          int, number of output channels (d_model).
        target_size:      tuple (H, W) the resolution to which all inputs are resized.
    """
    def __init__(self,
                 in_channels_list: list[int],
                 out_dim: int,
                 target_size: tuple[int, int]):
        super().__init__()
        self.target_size = target_size  # (H, W)
        total_in = sum(in_channels_list)
        self.fuse_conv = nn.Conv2d(total_in, out_dim, kernel_size=1)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        H, W = self.target_size
        resized = [
            F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            for f in features
        ]
        x = torch.cat(resized, dim=1)    # [B, sum(C_i), H, W]
        return self.fuse_conv(x)         # [B, out_dim, H, W]


from .block import ViLFusionBlock
# -------------------------------------------------------------------------- #
# ViL-based RT-DETR head with optional single-scale fusion
# -------------------------------------------------------------------------- #
class ViLBasedRTDETRHead(nn.Module):
    """
    RT-DETR detection head powered by ViL blocks.
    Can operate on multi-scale FPN outputs or a single fused scale.
    """
    def __init__(self,
                 ch_vil: list[int],
                 memory_spatial_shapes: list[list[int,int]],
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 num_vil_layers: int = 6,
                 vil_block_decoder_args: dict = None,
                 act=nn.SiLU,
                 nc: int = 80,
                 use_single_scale: bool = False,
                 fusion_target_size: list[int,int] = [20,20],
                 num_denoising: int = 100,
                 label_noise_ratio: float = 0.5,
                 box_noise_scale: float = 1.0,
                 grid_size_anchor: float = 0.05,
                 learnt_init_query: bool = False,
                 query_pos_head_num_layers: int = 2,
                 bbox_head_num_layers: int = 3,
                 return_intermediate_outputs: bool = True):
        super().__init__()
        self.use_single_scale = use_single_scale


        # fusion config

        if self.use_single_scale:
            in_channels_fused = sum(ch_vil)
            self.fuser = ViLFusionBlock(
                in_channels=in_channels_fused,
                hidden_dim=hidden_dim,
                config=dict(
                    seqlens=fusion_target_size,
                    chunk_size=512,
                    qkv_block_size=32,
                    mlp_ratio=4.0,
                    conv_kind="2d"
                ),
                n=1,
                mlp_ratio=4.0
            )
            ch_vil = [hidden_dim]  # Post-fusion channel dimension
            self.memory_spatial_shapes = [fusion_target_size]
        else:
            self.memory_spatial_shapes = [tuple(map(int, s)) for s in memory_spatial_shapes]

        self.hidden_dim   = hidden_dim
        self.num_queries  = num_queries
        self.num_vil_layers = num_vil_layers
        self.return_intermediate_outputs = return_intermediate_outputs
        self.nc           = nc
        self.grid_size_anchor = grid_size_anchor

        # per-level projection
        self.scale_proj = nn.ModuleList(nn.Conv2d(c, hidden_dim, 1) for c in ch_vil)
        self.scale_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)

        # positional + level embeddings
        self.pos_embed_tables = nn.ParameterList([
            nn.Parameter(_build_2d_sincos_embedding(hidden_dim, h, w), requires_grad=False)
            for (h, w) in self.memory_spatial_shapes
        ])
        self.level_embed = nn.Parameter(torch.zeros(len(self.memory_spatial_shapes), hidden_dim))
        self.register_buffer('vil_fpn_spatial_shapes',
                             torch.tensor(self.memory_spatial_shapes, dtype=torch.long),
                             persistent=False)

        # encoder proposal heads
        self.enc_output     = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head  = MLP(hidden_dim, hidden_dim, 4,
                                  num_layers=bbox_head_num_layers, act=nn.SiLU)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale   = box_noise_scale
        self.denoising_class_embed = (
            nn.Embedding(nc, hidden_dim) if num_denoising else None
        )

        # query embeddings
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2*hidden_dim, hidden_dim,
                                  num_layers=query_pos_head_num_layers, act=nn.SiLU)
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)

        # writer/reader
        base_cfg = dict(dim=hidden_dim, expansion=2, qkv_block_size=max(1, hidden_dim//64))
        if vil_block_decoder_args:
            base_cfg.update(vil_block_decoder_args)
        self.writer = nn.ModuleList(
            ViLLayerFusedQKV(**base_cfg,
                             direction=(SequenceTraversal.ROWWISE_FROM_TOP_LEFT
                                        if i%2==0 else SequenceTraversal.ROWWISE_FROM_BOT_RIGHT))
            for i in range(num_vil_layers)
        )
        dir_pairs = _reader_traversal_pairs(num_vil_layers, SequenceTraversal, mem_flip_every=1, qry_flip_every=2)
        self.reader = nn.ModuleList(
            RefactoredxLSTMCrossReader(dim=hidden_dim,
                                       expansion=base_cfg['expansion'],
                                       qkv_block_size=base_cfg['qkv_block_size'],
                                       seqlens_list=self.memory_spatial_shapes,
                                       memory_direction=m_dir,
                                       query_direction=q_dir)
            for m_dir, q_dir in dir_pairs
        )

        # prediction heads
        heads = num_vil_layers if return_intermediate_outputs else 1
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, nc) for _ in range(heads))
        self.dec_bbox_head  = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4,
                num_layers=bbox_head_num_layers, act=nn.SiLU)
            for _ in range(heads)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) * sqrt(self.hidden_dim/256)
        xavier_uniform_(self.enc_output[0].weight)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias,   0.0)
        constant_(self.enc_score_head.bias, bias_cls)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        if hasattr(self.query_pos_head.layers[0], 'weight'):
            xavier_uniform_(self.query_pos_head.layers[0].weight)
        for cls_h, reg_h in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_h.bias, bias_cls)
            constant_(reg_h.layers[-1].weight, 0.0)
            constant_(reg_h.layers[-1].bias,   0.0)
        nn.init.normal_(self.level_embed, std=0.02)
        for proj in self.scale_proj:
            xavier_uniform_(proj.weight)
            if proj.bias is not None:
                constant_(proj.bias, 0.0)

    def _generate_anchors(self, shapes, B, dtype, device, eps=1e-2):
        anchors, valid = [], []
        for lvl, (H, W) in enumerate(shapes):
            gy, gx = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                                    torch.arange(W, dtype=dtype, device=device),
                                    indexing='ij')
            grid = torch.stack([gx, gy], -1).unsqueeze(0)
            grid = (grid + 0.5) / torch.tensor([W, H], dtype=dtype, device=device)
            wh   = torch.ones_like(grid) * self.grid_size_anchor * (2.0 ** lvl)
            a    = torch.cat([grid, wh], -1).view(1, -1, 4)
            anchors.append(a)
            valid.append(((a > eps) & (a < 1 - eps)).all(-1, keepdim=True))
        anchors = torch.cat(anchors, 1)
        valid   = torch.cat(valid,   1)
        anchors = torch.log(anchors / (1 - anchors)).masked_fill(~valid, float('inf'))
        return anchors.repeat(B, 1, 1), valid.repeat(B, 1, 1)

    def _initial_queries(self, feats, shapes, batch):
        B, N, _ = feats.shape
        anchors, valid = self._generate_anchors(shapes, B, feats.dtype, feats.device)
        enc_feats  = self.enc_output(valid * feats)
        cls_logits = self.enc_score_head(enc_feats)
        box_logit  = self.enc_bbox_head(enc_feats) + anchors
        k = min(self.num_queries, N)
        topk = torch.topk(cls_logits.max(-1).values, k, dim=1).indices
        gather = lambda src, d: torch.gather(src, 1,
                                             topk.unsqueeze(-1).expand(-1, -1, d))
        sel_feat  = gather(enc_feats, self.hidden_dim)
        sel_box   = gather(box_logit, 4)
        sel_score = gather(cls_logits, self.nc)
        dn_c, dn_b, _, dn_meta = get_cdn_group(batch, self.nc, k,
                                                self.denoising_class_embed.weight if self.denoising_class_embed else None,
                                                self.num_denoising, self.label_noise_ratio,
                                                self.box_noise_scale, self.training)
        base_q = self.tgt_embed.weight.unsqueeze(0).repeat(B,1,1) if self.learnt_init_query else sel_feat
        base_b = sel_box
        if self.training:
            if not self.learnt_init_query:
                base_q = base_q
            base_b = base_b
        if dn_c is not None and dn_b is not None:
            q_content = torch.cat([dn_c, base_q], 1)
            q_boxes   = torch.cat([dn_b, base_b], 1)
        else:
            q_content, q_boxes = base_q, base_b
            if dn_meta:
                dn_meta['num_denoising_queries'] = 0
                dn_meta['dn_num_split'][0]       = 0
        dn_len = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        if self.training and dn_len > 0:
            zeros     = torch.zeros(B, dn_len, self.nc, device=cls_logits.device, dtype=cls_logits.dtype)
            enc_scores= torch.cat([zeros, sel_score], 1)
            enc_boxes = torch.cat([dn_b.sigmoid(), sel_box.sigmoid()], 1)
        else:
            enc_scores, enc_boxes = sel_score, sel_box.sigmoid()
        return q_content, q_boxes, enc_boxes, enc_scores, dn_meta

    @torch.compile
    def forward(self, feats_per_level, batch=None):
        B = feats_per_level[0].size(0)
        if self.use_single_scale:
            H, W = self.memory_spatial_shapes[0]
            resized_feats = [
                F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
                for f in feats_per_level
            ]
            x = torch.cat(resized_feats, dim=1)  # [B, sum(C), H, W]
            fused = self.fuser(x)                # ViLFusionBlock expects single input

            x = self.scale_proj[0](fused)        # project to hidden_dim if needed
            x = x + self.pos_embed_tables[0].unsqueeze(0).to(x.dtype)
            x = x + self.level_embed[0].view(1, -1, 1, 1)
            x = self.scale_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            memory = x.reshape(B, -1, self.hidden_dim)

        else:
            mem_parts = []
            for lvl, feat in enumerate(feats_per_level):
                x = self.scale_proj[lvl](feat)
                x = x + self.pos_embed_tables[lvl].unsqueeze(0).to(x.dtype)
                x = x + self.level_embed[lvl].view(1, -1, 1, 1)
                x = self.scale_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
                mem_parts.append(x.reshape(B, -1, self.hidden_dim))
            memory = torch.cat(mem_parts, 1)
        q_cnt, q_box, enc_boxes, enc_scores, dn_meta = \
            self._initial_queries(memory, self.memory_spatial_shapes, batch)
        queries = self.decoder_norm(q_cnt + self.query_pos_head(q_box.sigmoid()))
        dn_len  = dn_meta.get('num_denoising_queries', 0) if dn_meta else 0
        dec_scores, dec_boxes = [], []
        cur_q, cur_box = queries, q_box
        for i in range(self.num_vil_layers):
            if dn_len:
                cur_q = torch.cat([self.writer[i](cur_q[:, :dn_len]),
                                   self.writer[i](cur_q[:, dn_len:])], 1)
            else:
                cur_q = self.writer[i](cur_q)
            cur_q = self.reader[i](memory, cur_q)
            head_idx = i if self.return_intermediate_outputs else 0
            cls_log   = self.dec_score_head[head_idx](cur_q)
            cur_box   = cur_box + self.dec_bbox_head[head_idx](cur_q)
            dec_scores.append(cls_log)
            dec_boxes .append(cur_box.sigmoid())
        dec_scores = torch.stack(dec_scores)
        dec_boxes  = torch.stack(dec_boxes)
        if self.training:
            return dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta
        final_scores, final_boxes = dec_scores[-1], dec_boxes[-1]
        if dn_len:
            final_scores, final_boxes = final_scores[:, dn_len:], final_boxes[:, dn_len:]
        det_out = torch.cat([final_boxes, final_scores.sigmoid()], -1)
        return det_out, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)

# -------------------------------------------------------------------------- #
# Note: ViLBasedRTDETRHead is agnostic to scale count; fusion flag toggles.   #
# -------------------------------------------------------------------------- #
