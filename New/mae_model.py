import torch
import torch.nn as nn
from hovertrans import create_model
from config import config


class MAEHover(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = create_model(
            img_size=config.img_size,
            num_classes=config.class_num,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            num_inner_head=config.num_inner_head
        )

        # Remove classification head
        self.encoder.head = nn.Identity()

        embed_dim = config.dim[-1] * 2

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 256 * 3)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        features = self.encoder.avgpool(features).flatten(1)
        features = self.encoder.norm(features)

        recon = self.decoder(features)
        recon = recon.view(x.size(0), 3, 256, 256)

        return recon
