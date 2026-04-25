import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Bloc convolutie 3D: Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN3D(nn.Module):
    """
    Retea convolutionala 3D pentru clasificare binara noduli pulmonari.

    Parametri
    ----------
    dropout : float
        Probabilitate dropout in stratul fully connected (default: 0.3)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock3D(1,  16),   # [B, 1, 64, 64, 64] → [B, 16, 32, 32, 32]
            ConvBlock3D(16, 32),   # → [B, 32, 16, 16, 16]
            ConvBlock3D(32, 64),   # → [B, 64,  8,  8,  8]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # → [B, 64, 1, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),           # → [B, 64]
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),           # probabilitate in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x.squeeze(1)  # [B, 1] → [B]

    def count_parameters(self) -> int:
        """Returneaza numarul total de parametri antrenabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
