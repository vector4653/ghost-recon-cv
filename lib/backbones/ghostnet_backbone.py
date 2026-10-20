import torch
import torch.nn as nn
import timm


class GhostNetBackbone(nn.Module):
    """
    GhostNet backbone wrapper using timm's ghostnet_100.
    Extracts multi-scale stage outputs matching SINet-V2 decoder requirements:
      - x1: stride 4  (88x88),  channels=24
      - x2: stride 8  (44x44),  channels=40
      - x3: stride 16 (22x22),  channels=112
      - x4: stride 32 (11x11),  channels=160
    """
    def __init__(self, pretrained=True):
        super(GhostNetBackbone, self).__init__()
        # Load timm ghostnet_100 with optional ImageNet pretrained weights
        self.model = timm.create_model('ghostnet_100', pretrained=pretrained)
        self.conv_stem = self.model.conv_stem
        self.bn1 = self.model.bn1
        self.act1 = self.model.act1
        self.blocks = self.model.blocks

    def forward(self, x):
        # Stem: 352x352 -> 176x176 (stride 2)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Block 0: stride 2 (176x176, 16 channels)
        x = self.blocks[0](x)

        # Blocks 1-2: stride 4 (88x88, 24 channels)
        x1 = self.blocks[1](x)
        x1 = self.blocks[2](x1)

        # Blocks 3-4: stride 8 (44x44, 40 channels)
        x2 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)

        # Blocks 5-6: stride 16 (22x22, 112 channels)
        x3 = self.blocks[5](x2)
        x3 = self.blocks[6](x3)

        # Blocks 7-8: stride 32 (11x11, 160 channels)
        x4 = self.blocks[7](x3)
        x4 = self.blocks[8](x4)

        return x1, x2, x3, x4


def ghostnet_100(pretrained=True):
    return GhostNetBackbone(pretrained=pretrained)
