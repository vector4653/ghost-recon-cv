import torch
import torch.nn as nn
import torchvision.models as models


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 1.0x backbone wrapper using torchvision.
    Extracts multi-scale stage outputs matching SINet-V2 decoder requirements:
      - x1: stride 4  (88x88),  channels=24
      - x2: stride 8  (44x44),  channels=116
      - x3: stride 16 (22x22),  channels=232
      - x4: stride 32 (11x11),  channels=464
    """
    def __init__(self, pretrained=True):
        super(ShuffleNetV2Backbone, self).__init__()
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        base = models.shufflenet_v2_x1_0(weights=weights)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4

    def forward(self, x):
        # conv1: 352x352 -> 176x176 (stride 2)
        x = self.conv1(x)
        # maxpool: 176x176 -> 88x88 (stride 4, 24 channels)
        x1 = self.maxpool(x)
        # stage2: 88x88 -> 44x44 (stride 8, 116 channels)
        x2 = self.stage2(x1)
        # stage3: 44x44 -> 22x22 (stride 16, 232 channels)
        x3 = self.stage3(x2)
        # stage4: 22x22 -> 11x11 (stride 32, 464 channels)
        x4 = self.stage4(x3)

        return x1, x2, x3, x4


def shufflenet_v2_x1_0(pretrained=True):
    return ShuffleNetV2Backbone(pretrained=pretrained)
