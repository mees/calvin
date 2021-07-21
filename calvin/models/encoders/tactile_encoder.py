import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class TactileEncoder(nn.Module):
    def __init__(self, freeze_tactile_backbone=True):
        super(TactileEncoder, self).__init__()
        # Load pre-trained resnet-18
        net = models.resnet18(pretrained=True)
        # Remove the last fc layer, and rebuild
        modules = list(net.children())[:-1]
        self.net = nn.Sequential(*modules)
        if freeze_tactile_backbone:
            for param in self.net.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # batch, 512, 1, 1
        # Add fc layer for final prediction
        x = torch.flatten(x, start_dim=1)  # batch, 512
        output = F.relu(self.fc1(x))  # batch, 256
        output = self.fc2(output)  # batch, 64
        return output
