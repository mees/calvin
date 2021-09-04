#!/usr/bin/env python3

import math
from typing import Dict, Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F


class PlanRecognitionNetwork(nn.Module):
    def __init__(
        self,
        visual_features: int,
        plan_features: int,
        n_state_obs: int,
        action_space: int,
        birnn_dropout_p: float,
        min_std: float,
    ):
        super(PlanRecognitionNetwork, self).__init__()
        self.visual_features = visual_features
        self.plan_features = plan_features
        self.n_state_obs = n_state_obs
        self.action_space = action_space
        self.min_std = min_std
        self.in_features = self.visual_features + self.n_state_obs
        self.birnn_model = nn.RNN(
            input_size=self.in_features,
            hidden_size=2048,
            nonlinearity="relu",
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=birnn_dropout_p,
        )  # shape: [N, seq_len, 64+8]
        self.mean_fc = nn.Linear(in_features=4096, out_features=self.plan_features)  # shape: [N, seq_len, 4096]
        self.variance_fc = nn.Linear(in_features=4096, out_features=self.plan_features)  # shape: [N, seq_len, 4096]

    def forward(self, perceptual_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, hn = self.birnn_model(perceptual_emb)
        x = x[:, -1]  # we just need only last unit output
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std  # shape: [N, 256]

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pr_dist = Independent(Normal(mean, std), 1)
        return pr_dist
