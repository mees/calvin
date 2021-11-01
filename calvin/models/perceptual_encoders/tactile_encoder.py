import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class TactileEncoder(nn.Module):
    def __init__(self, visual_features: int, freeze_tactile_backbone: bool = True):
        super(TactileEncoder, self).__init__()
        # Load pre-trained resnet-18
        net = models.resnet18(pretrained=True)
        # Remove the last fc layer, and rebuild
        modules = list(net.children())[:-1]
        self.net = nn.Sequential(*modules)
        if freeze_tactile_backbone:
            for param in self.net.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = self.net(x[:, :3, :, :]).squeeze()
        x_r = self.net(x[:, 3:, :, :]).squeeze()
        x = torch.cat((x_l, x_r), dim=-1)
        # Add fc layer for final prediction
        output = F.relu(self.fc1(x))  # batch, 512
        output = self.fc2(output)  # batch, 64
        return output
