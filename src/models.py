import torch
import torch.nn as nn
import numpy as np

class FusionBase(nn.Module):
    """Base class to handle different fusion strategies."""
    def fuse(self, x1, x2, strategy):
        if strategy == "concat":
            return torch.cat((x1, x2), dim=1)
        elif strategy == "add":
            return x1 + x2
        elif strategy == "hadamard":
            return x1 * x2
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

class LateFusionModel(FusionBase):
    def __init__(self):
        super().__init__()

        self.rgb_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16, 100), nn.ReLU()
        )

        self.lidar_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16, 100), nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(200, 100), 
            nn.ReLU(), 
            nn.Linear(100, 2)
        )

    def forward(self, rgb, lidar):
        rgb_feat = self.rgb_net(rgb)
        lidar_feat = self.lidar_net(lidar)
        combined = torch.cat((rgb_feat, lidar_feat), dim=1)
        return self.head(combined)

class IntermediateFusionModel(FusionBase):
    def __init__(self, fusion="concat", use_strided=False):
        super().__init__()
        self.fusion = fusion

        def make_block(in_ch, out_ch):
            if use_strided:

                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.ReLU()
                )
            else:

                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

        self.rgb_early = nn.Sequential(
            make_block(4, 32),
            make_block(32, 64)
        )
        self.lidar_early = nn.Sequential(
            make_block(4, 32),
            make_block(32, 64)
        )

        mid_ch = 128 if fusion == "concat" else 64

        self.shared = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(mid_ch, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten(), 
            nn.Linear(128*4*4, 2)
        )

    def forward(self, rgb, lidar):
        x1 = self.rgb_early(rgb)
        x2 = self.lidar_early(lidar)
        fused = self.fuse(x1, x2, self.fusion)
        return self.shared(fused)

class CILPModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.rgb_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, embed_dim)
        )

        self.lidar_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, embed_dim)
        )
 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, rgb, lidar):
        rgb_feats = self.rgb_net(rgb)
        lidar_feats = self.lidar_net(lidar)

        rgb_feats = rgb_feats / rgb_feats.norm(dim=1, keepdim=True)
        lidar_feats = lidar_feats / lidar_feats.norm(dim=1, keepdim=True)
        
        return rgb_feats, lidar_feats, self.logit_scale.exp()

class CrossModalProjector(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)