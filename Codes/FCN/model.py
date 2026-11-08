"""FCN-8s (Long et al., 2015) with a VGG16 encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class FCN8s(nn.Module):
    """Fully Convolutional Network with 8x upsampling.

    The VGG16 classifier is discarded; the convolutional features are used as
    the encoder. Skip connections from pool3 (stride 8) and pool4 (stride 16)
    are added before the final 8x upsampling.
    """

    def __init__(self, num_classes=2, pretrained_backbone=False):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        features = vgg16(weights=weights).features

        # VGG16 features: pool3 ends at index 16, pool4 at 23, pool5 at 30.
        self.pool3 = features[:17]    # 256 channels, stride 8
        self.pool4 = features[17:24]  # 512 channels, stride 16
        self.pool5 = features[24:]    # 512 channels, stride 32
        self.dropout = nn.Dropout2d(p=0.1)

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)

        # 2x upsampling (kernel 4, stride 2) and final 8x upsampling.
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        for layer in (self.score_pool3, self.score_pool4, self.score_pool5):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)
        for layer in (self.upscore2, self.upscore4, self.upscore8):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        input_size = x.shape[-2:]

        p3 = self.pool3(x)          # H/8
        p4 = self.pool4(p3)         # H/16
        p5 = self.pool5(p4)         # H/32
        p5 = self.dropout(p5)

        s5 = self.score_pool5(p5)
        s4 = self.score_pool4(p4)
        up5 = self.upscore2(s5)
        if up5.shape[-2:] != s4.shape[-2:]:
            up5 = F.interpolate(up5, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        s4 = s4 + up5

        s3 = self.score_pool3(p3)
        up4 = self.upscore4(s4)
        if up4.shape[-2:] != s3.shape[-2:]:
            up4 = F.interpolate(up4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = s3 + up4

        out = self.upscore8(s3)

        # Keeps the output exactly the same size as the input for odd sizes.
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out
