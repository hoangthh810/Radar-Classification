import torch
import torch.nn as nn

from models.common import DynamicActivation, SEBlock, DualAttn, InvResAttentionBlock

class SOTACNN(nn.Module):
    """
    SOTACNN — Lightweight CNN (<100K params) for image classification.

    Architecture overview:
       Backbone: 4-stage InvResAttentionBlock with linearly increasing stochastic depth.
       Head:  GAP + GMP fusion from stage3 & stage4 → Linear classifier.

    Key components:
      - DynamicActivation : learnable per-channel activation (replaces ReLU)
      - SEBlock           : Squeeze-and-Excitation channel attention
      - DualAttn          : strip spatial attention (H & W) + SE channel attention
      - InvResAttentionBlock : Inverted Residual Block with DualAttn + Stochastic Depth
    """

    def __init__(self, num_classes: int = 12):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            DynamicActivation(16),
        )
        self.stage1 = nn.Sequential(
            InvResAttentionBlock(16, 24, stride=2, expand_ratio=2, dilation=1),
            InvResAttentionBlock(24, 24, stride=1, expand_ratio=2, dilation=1),
        )
        self.stage2 = nn.Sequential(
            InvResAttentionBlock(24, 32, stride=2, expand_ratio=2, dilation=1),
            InvResAttentionBlock(32, 32, stride=1, expand_ratio=2, dilation=1),
        )
        self.stage3 = nn.Sequential(
            InvResAttentionBlock(32, 48, stride=2, expand_ratio=2, dilation=1),
            InvResAttentionBlock(48, 48, stride=1, expand_ratio=2, dilation=2),
        )
        self.stage4 = nn.Sequential(
            InvResAttentionBlock(48, 64, stride=2, expand_ratio=2, dilation=1),
            InvResAttentionBlock(64, 64, stride=1, expand_ratio=2, dilation=2),
        )

        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.gmp        = nn.AdaptiveMaxPool2d(1)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(48 + 48 + 64 + 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stem(x)
        x  = self.stage1(x)
        x  = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)

        f3 = torch.cat([self.gap(x3), self.gmp(x3)], dim=1).flatten(1)
        f4 = torch.cat([self.gap(x4), self.gmp(x4)], dim=1).flatten(1)
        f  = self.dropout(torch.cat([f3, f4], dim=1))

        return self.classifier(f)