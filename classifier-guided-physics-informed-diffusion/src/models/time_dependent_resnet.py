import math
import torch
import torchvision
import torch.nn as nn

def timestep_embedding(timesteps, dim):
    # sinusoidal timestep embedding
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class TimeDependentResNet(nn.Module):
    def __init__(self, num_classes, time_dim=128, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        pretrained_conv1_weight = resnet.conv1.weight.data.clone()  # [64, 3, 7, 7]
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average RGB channels so the pretrained features transfer to grayscale input
            resnet.conv1.weight.data = pretrained_conv1_weight.mean(dim=1, keepdim=True)
        modules = list(resnet.children())[:-1]  # remove final fc
        self.backbone = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features

        # Project timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, self.feature_dim),
            nn.ReLU()
        )

        # Classifier
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B,)
        feats = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, feature_dim)

        # timestep embedding
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)  # (B, feature_dim)

        # Add conditioning
        feats = feats + t_emb

        return self.fc(feats)