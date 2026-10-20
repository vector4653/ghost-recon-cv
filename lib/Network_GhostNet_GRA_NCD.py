import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbones.ghostnet_backbone import GhostNetBackbone
from lib.Network_Res2Net_GRA_NCD import BasicConv2d, RFB_modified, NeighborConnectionDecoder, ReverseStage


class Network(nn.Module):
    # GhostNet based encoder decoder for SINet-V2
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- GhostNet Backbone ----
        self.backbone = GhostNetBackbone(pretrained=imagenet_pretrained)
        
        # ---- Receptive Field Block like module ----
        # GhostNet stage channels: x2 -> 40, x3 -> 112, x4 -> 160
        self.rfb2_1 = RFB_modified(40, channel)
        self.rfb3_1 = RFB_modified(112, channel)
        self.rfb4_1 = RFB_modified(160, channel)
        
        # ---- Partial Decoder (NCD) ----
        self.NCD = NeighborConnectionDecoder(channel)

        # ---- Reverse Stages (GRA) ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        # Feature Extraction: returns x1 (stride 4), x2 (stride 8), x3 (stride 16), x4 (stride 32)
        _, x2, x3, x4 = self.backbone(x)

        # Receptive Field Block (enhanced) -> channel 32
        x2_rfb = self.rfb2_1(x2)        # (bs, 32, 44, 44)
        x3_rfb = self.rfb3_1(x3)        # (bs, 32, 22, 22)
        x4_rfb = self.rfb4_1(x4)        # (bs, 32, 11, 11)

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred
