#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class VisionNetwork(nn.Module):
    # reference: https://arxiv.org/pdf/2005.07648.pdf
    def __init__(
        self,
        input_width: int,
        input_height: int,
        activation_function: str,
        dropout_vis_fc: float,
        l2_normalize_output: bool,
        visual_features: int,
        num_c: int,
    ):
        super(VisionNetwork, self).__init__()
        self.l2_normalize_output = l2_normalize_output
        self.act_fn = getattr(nn, activation_function)()
        # w,h,kernel_size,padding,stride
        w, h = self.calc_out_size(input_width, input_height, 8, 0, 4)
        w, h = self.calc_out_size(w, h, 4, 0, 2)
        w, h = self.calc_out_size(w, h, 3, 0, 1)
        self.spatial_softmax = SpatialSoftmax(num_rows=w, num_cols=h, temperature=1.0)  # shape: [N, 128]
        # model
        self.conv_model = nn.Sequential(
            # input shape: [N, 3, 200, 200]
            nn.Conv2d(in_channels=num_c, out_channels=32, kernel_size=8, stride=4),  # shape: [N, 32, 49, 49]
            self.act_fn,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # shape: [N, 64, 23, 23]
            self.act_fn,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # shape: [N, 64, 21, 21]
            self.act_fn,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=512), self.act_fn, nn.Dropout(dropout_vis_fc)
        )  # shape: [N, 512]
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)  # shape: [N, 64]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)
        x = self.spatial_softmax(x)
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


class SpatialSoftmax(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, temperature: Optional[float] = None):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
        :param num_rows:  size related to original image width
        :param num_cols:  size related to original image height
        :param temperature: Softmax temperature (optional). If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, num_cols), torch.linspace(-1.0, 1.0, num_rows), indexing="ij"
        )
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        if temperature:
            self.register_buffer("temperature", torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(-1, h * w)  # batch, C, W*H
        softmax_attention = F.softmax(x / self.temperature, dim=1)  # batch, C, W*H
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords  # batch, C*2
