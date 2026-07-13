import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # GroupNorm (not BatchNorm) -- normalizes within each image, no batch
        # statistics/running average, so no train/eval mismatch. BatchNorm was
        # causing huge validation-loss spikes: an unusual batch (size 16) would
        # briefly corrupt its running stats, and every eval pass until the next
        # recovery used those corrupted stats.
        self.features = nn.Sequential(
            # Block 1: 1×150×150 → 16×75×75
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16×75×75 → 32×37×37
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32×37×37 → 64×18×18
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.6)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
