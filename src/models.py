import torch
import torch.nn as nn
import torch.nn.functional as F


# ── v1: TinyCNN ──────────────────────────────────────────────────────────────
# 2 conv layers, ~50K params. Expected to UNDERFIT — too little capacity for 7
# emotion classes at 48x48. Used as the intentional underfitting baseline.
class TinyCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # → (8, 48, 48)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (8, 24, 24)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # → (16, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (16, 12, 12)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * 12 * 12, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── v2: SmallCNN ─────────────────────────────────────────────────────────────
# 4 conv layers, no BatchNorm, no Dropout. ~1.2M params.
# High capacity with no regularization → expected to OVERFIT.
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),   # 24x24
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),   # 12x12
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 6x6
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(), nn.MaxPool2d(2),   # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── v3: MediumCNN ────────────────────────────────────────────────────────────
# 4 conv blocks with BatchNorm + Dropout. Regularized version of SmallCNN.
# BatchNorm stabilises training; Dropout reduces co-adaptation of neurons.
class MediumCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            if pool:
                layers += [nn.MaxPool2d(2), nn.Dropout2d(0.25)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1, 32),    # → (32, 24, 24)
            conv_block(32, 64),   # → (64, 12, 12)
            conv_block(64, 128),  # → (128, 6, 6)
            conv_block(128, 256), # → (256, 3, 3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── v4: ResNetStyleCNN ───────────────────────────────────────────────────────
# Custom residual blocks. Skip connections allow gradients to bypass layers,
# enabling deeper networks without vanishing gradients. Good inductive bias
# for learning hierarchical facial features.
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.dropout(self.relu(x + self.block(x)))


class ResNetStyleCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, dropout=0.1),  nn.MaxPool2d(2))  # 24
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64),
            ResidualBlock(64, dropout=0.1), nn.MaxPool2d(2),                            # 12
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128),
            ResidualBlock(128, dropout=0.1), nn.MaxPool2d(2),                           # 6
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256),
            ResidualBlock(256, dropout=0.1), nn.MaxPool2d(2),                           # 3
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(x)


# ── v5: TransferLearningCNN ──────────────────────────────────────────────────
# MobileNetV2 pretrained on ImageNet. Grayscale images are resized to 224x224
# and converted to 3-channel (expected by dataset.py transform).
# Early layers capture universal features (edges, textures); we fine-tune the
# top layers for emotion-specific representations.
class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5, freeze_until=-3):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze all feature layers first
        for param in base.features.parameters():
            param.requires_grad = False

        # Unfreeze the last |freeze_until| feature blocks
        if freeze_until < 0:
            for param in base.features[freeze_until:].parameters():
                param.requires_grad = True

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ── Registry ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    'tiny_cnn':         TinyCNN,
    'small_cnn':        SmallCNN,
    'medium_cnn':       MediumCNN,
    'resnet_style':     ResNetStyleCNN,
    'transfer_learning': TransferLearningCNN,
}


def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
