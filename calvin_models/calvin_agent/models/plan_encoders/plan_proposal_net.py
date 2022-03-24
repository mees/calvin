#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F


class PlanProposalNetwork(nn.Module):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        activation_function: str,
        min_std: float,
    ):
        super(PlanProposalNetwork, self).__init__()
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.min_std = min_std
        self.in_features = self.perceptual_features + self.latent_goal_features
        self.act_fn = getattr(nn, activation_function)()
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=2048),  # shape: [N, 136]
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
        )
        self.mean_fc = nn.Linear(in_features=2048, out_features=self.plan_features)  # shape: [N, 2048]
        self.variance_fc = nn.Linear(in_features=2048, out_features=self.plan_features)  # shape: [N, 2048]

    def forward(self, initial_percep_emb: torch.Tensor, latent_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([initial_percep_emb, latent_goal], dim=-1)
        x = self.fc_model(x)
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std  # shape: [N, 256]

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pp_dist = Independent(Normal(mean, std), 1)
        return pp_dist
