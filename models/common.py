import torch
import torch.nn as nn


class DynamicActivation(nn.Module):
    """Learnable activation parameterized per channel.

    Generalizes Swish: f(x) = (p1 - p2) * x * sigmoid(beta * x) + p2 * x
    """

    def __init__(self, width: int):
        super().__init__()
        self.p1   = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2   = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.p1 - self.p2) * x * torch.sigmoid(self.beta * x) + self.p2 * x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DualAttn(nn.Module):
    """Dual Attention: strip spatial attention (H & W) + SE channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.Hpool    = nn.AdaptiveAvgPool2d((None, 1))
        self.Vpool    = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h   = nn.Conv2d(channels, mid, (3, 1), padding=(1, 0))
        self.conv_w   = nn.Conv2d(channels, mid, (3, 1), padding=(1, 0))
        self.conv_fuse = nn.Conv2d(mid, channels, 1)
        self.sigmoid  = nn.Sigmoid()
        self.se       = SEBlock(channels, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_h = self.conv_h(self.Hpool(x))
        x_w = self.Vpool(x).permute(0, 1, 3, 2)
        x_w = self.conv_w(x_w).permute(0, 1, 3, 2)
        s_att = self.sigmoid(self.conv_fuse(x_h + x_w))
        return self.se(x * s_att)


class InvResAttentionBlock(nn.Module):
    """Inverted Residual Block with DualAttn."""

    def __init__(
            self,
            in_ch: int, out_ch: int,
            stride: int = 1, expand_ratio: int = 2,
            dilation: int = 1,
    ):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        hidden_dim = int(round(in_ch * expand_ratio))

        # 1. Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            DynamicActivation(hidden_dim)
        ) if expand_ratio != 1 else nn.Identity()

        # 2. Depthwise Conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride,
                padding=dilation, dilation=dilation,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            DynamicActivation(hidden_dim)
        )

        # 3. Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # 4. Attention
        self.att = DualAttn(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.expand(x)
        branch = self.depthwise(branch)
        branch = self.project(branch)
        branch = self.att(branch)

        # Residual connection tiêu chuẩn (Không có Drop Path)
        if self.use_res_connect:
            return x + branch
        return branch