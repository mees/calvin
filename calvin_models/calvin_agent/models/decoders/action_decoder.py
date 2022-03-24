from typing import Tuple

import torch
from torch import nn


class ActionDecoder(nn.Module):
    def act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
