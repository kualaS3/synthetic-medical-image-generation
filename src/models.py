"""Models: conditional Generator, Discriminator, and downstream classifier.

``weights_init``, ``Generator`` and ``Discriminator`` are verbatim from the
notebook (cells 14-16). ``DRClassifier`` is verbatim from cell 29.
"""
import torch
import torch.nn as nn
from torchvision import models


def weights_init(m):
    """
    Initialise weights from the DCGAN paper (Radford et al., 2015).
    Conv weights: N(0, 0.02), BatchNorm weights: N(1, 0.02), biases: 0.
    This specific initialisation is important for stable GAN training.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Conditional Generator: noise z + grade label c → 128x128 RGB fundus image.

    Architecture:
    - Grade label is embedded (learnable) and concatenated with z
    - 5 transposed convolution blocks, each doubling spatial resolution
    - 1x1 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128
    - BatchNorm + ReLU after each block, Tanh at the end (output in [-1, 1])

    This maps to the 'Input Concatenation' conditioning strategy from
    slide 57. The condition c is concatenated with input z.
    """

    def __init__(self, latent_dim, n_classes, img_size, ngf=64):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        input_dim = latent_dim + n_classes

        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # → ngf*16 × 4 × 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # → ngf*8 × 8 × 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # → ngf*4 × 16 × 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # → ngf*2 × 32 × 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # → ngf × 64 × 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # → 3 × 128 × 128
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)  # (B, n_classes)
        gen_input = torch.cat([z, label_input], dim=1)  # (B, latent+n_classes)
        gen_input = gen_input.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return self.net(gen_input)


class Discriminator(nn.Module):
    """
    Conditional Discriminator: (image, grade label) → real/fake probability.

    Architecture:
    - Grade label is embedded to a small 16×16 map, then bilinearly upsampled
      to match img_size, which keeps it resolution-agnostic and memory-efficient
    - 5 strided convolution blocks, each halving spatial resolution
    - LeakyReLU throughout, Sigmoid at the end
    """

    def __init__(self, n_classes, img_size, ndf=64):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(n_classes, 16 * 16)  # fixed small map

        self.net = nn.Sequential(
            # Input: 4 channels (3 RGB + 1 upsampled label map)
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(ndf * 16, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels)  # (B, 256)
        label_map = label_map.view(img.size(0), 1, 16, 16)  # (B, 1, 16, 16)
        label_map = torch.nn.functional.interpolate(
            label_map, size=(img.size(2), img.size(3)), mode="bilinear", align_corners=False
        )  # (B, 1, H, W)
        d_input = torch.cat([img, label_map], dim=1)  # (B, 4, H, W)
        return self.net(d_input).view(img.size(0), -1).mean(dim=1)


class DRClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for 5-class DR grading.
    Using transfer learning from ImageNet, a pragmatic choice given the
    limited training data, especially for Grades 3 and 4.
    """

    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
