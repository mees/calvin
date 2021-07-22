#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def nature_cnn(act_fn):
    return nn.Sequential(
        nn.Conv2d(3, 32, 8, stride=4),
        act_fn,
        nn.Conv2d(32, 64, 4, stride=2),
        act_fn,
        nn.Conv2d(64, 64, 3, stride=1),
        act_fn,
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 7 * 7, 128),
        act_fn,
    )


def cnn_3_layers(act_fn):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2),
        act_fn,
        nn.Conv2d(32, 32, 3, stride=2),
        act_fn,
        nn.Conv2d(32, 32, 3, stride=2),
        act_fn,
        nn.Flatten(start_dim=1),
        nn.Linear(32 * 9 * 9, 128),
    )


def cnn_4_layers(act_fn):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2),
        act_fn,
        nn.Conv2d(32, 32, 3, stride=2),
        act_fn,
        nn.Conv2d(32, 32, 3, stride=2),
        act_fn,
        nn.Conv2d(32, 32, 3, stride=1),
        act_fn,
        nn.Flatten(start_dim=1),
        nn.Linear(32 * 7 * 7, 128),
    )


class VisionNetwork(nn.Module):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        conv_encoder: str,
        activation_function: str,
        dropout_vis_fc: float,
        l2_normalize_output: bool,
        visual_features: int,
    ):
        super(VisionNetwork, self).__init__()
        self.l2_normalize_output = l2_normalize_output
        self.act_fn = getattr(nn, activation_function)()
        # model
        # this calls the method with the name conv_encoder
        self.conv_model = eval(conv_encoder)
        self.conv_model = self.conv_model(self.act_fn)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=512), self.act_fn, nn.Dropout(dropout_vis_fc)
        )  # shape: [N, 512]
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)  # shape: [N, 64]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x  # shape: [N, 64]

    @staticmethod
    def calc_out_size(w: int, h: int, kernel_size: int, padding: int, stride: int) -> Tuple[int, int]:
        width = (w - kernel_size + 2 * padding) // stride + 1
        height = (h - kernel_size + 2 * padding) // stride + 1
        return width, height
