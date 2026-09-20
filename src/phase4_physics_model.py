import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Phase4PhysicsResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        # --------------------------------------------------
        # Adapt first convolution: 3 RGB channels -> 2
        # physics channels
        # --------------------------------------------------

        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            # Average pretrained RGB filters
            averaged_weights = old_conv.weight.mean(
                dim=1,
                keepdim=True
            )

            # Use the averaged pretrained representation
            # for both physics channels.
            new_conv.weight.copy_(
                averaged_weights.repeat(1, 2, 1, 1)
            )

        self.backbone.conv1 = new_conv

        # --------------------------------------------------
        # Binary classifier
        # --------------------------------------------------

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):
        return self.backbone(x)