"""PyTorch replacements for the Detectron2 layers used by the v2 model."""

from dataclasses import dataclass
from typing import Optional

from torch import nn
from torch.nn import functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, *args, norm=None, activation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, inputs):
        outputs = F.conv2d(
            inputs, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )
        if self.norm is not None:
            outputs = self.norm(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def get_norm(norm, out_channels):
    if norm is None or norm == "":
        return None
    if not isinstance(norm, str):
        return norm(out_channels)
    factories = {
        "BN": nn.BatchNorm2d,
        "SyncBN": nn.SyncBatchNorm,
        "GN": lambda channels: nn.GroupNorm(32, channels),
    }
    if norm not in factories:
        raise ValueError(f"Unsupported normalization: {norm}")
    return factories[norm](out_channels)


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None