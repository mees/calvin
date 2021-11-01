from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualGoalEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_goal_features: int,
        in_features: int,
        l2_normalize_goal_embeddings: bool,
        activation_function: str,
    ):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=latent_goal_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x


class LanguageGoalEncoder(nn.Module):
    def __init__(
        self,
        language_features: int,
        hidden_size: int,
        latent_goal_features: int,
        word_dropout_p: float,
        l2_normalize_goal_embeddings: bool,
        activation_function: str,
    ):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Dropout(word_dropout_p),
            nn.Linear(in_features=language_features, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=latent_goal_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x
