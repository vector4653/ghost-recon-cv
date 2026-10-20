import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.Network_Res2Net_GRA_NCD import Network as Res2NetNetwork
from lib.Network_GhostNet_GRA_NCD import Network as GhostNetNetwork
from lib.Network_ShuffleNetV2_GRA_NCD import Network as ShuffleNetV2Network
from lib.backbones.ghostnet_backbone import GhostNetBackbone
from lib.backbones.shufflenetv2_backbone import ShuffleNetV2Backbone


class TestBackboneAndModelShapes(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.dummy_input = torch.randn(1, 3, 352, 352, device=self.device)

    def test_ghostnet_backbone_stages(self):
        backbone = GhostNetBackbone(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            x1, x2, x3, x4 = backbone(self.dummy_input)
        self.assertEqual(x1.shape, (1, 24, 88, 88), f"x1 expected (1, 24, 88, 88) got {x1.shape}")
        self.assertEqual(x2.shape, (1, 40, 44, 44), f"x2 expected (1, 40, 44, 44) got {x2.shape}")
        self.assertEqual(x3.shape, (1, 112, 22, 22), f"x3 expected (1, 112, 22, 22) got {x3.shape}")
        self.assertEqual(x4.shape, (1, 160, 11, 11), f"x4 expected (1, 160, 11, 11) got {x4.shape}")
        print("[PASS] GhostNetBackbone stage output shapes match expected: 24/40/112/160 at strides 4/8/16/32")

    def test_shufflenetv2_backbone_stages(self):
        backbone = ShuffleNetV2Backbone(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            x1, x2, x3, x4 = backbone(self.dummy_input)
        self.assertEqual(x1.shape, (1, 24, 88, 88), f"x1 expected (1, 24, 88, 88) got {x1.shape}")
        self.assertEqual(x2.shape, (1, 116, 44, 44), f"x2 expected (1, 116, 44, 44) got {x2.shape}")
        self.assertEqual(x3.shape, (1, 232, 22, 22), f"x3 expected (1, 232, 22, 22) got {x3.shape}")
        self.assertEqual(x4.shape, (1, 464, 11, 11), f"x4 expected (1, 464, 11, 11) got {x4.shape}")
        print("[PASS] ShuffleNetV2Backbone stage output shapes match expected: 24/116/232/464 at strides 4/8/16/32")

    def test_full_model_outputs_match_res2net(self):
        res2net_net = Res2NetNetwork(imagenet_pretrained=False).to(self.device).eval()
        ghostnet_net = GhostNetNetwork(imagenet_pretrained=False).to(self.device).eval()
        shufflenet_net = ShuffleNetV2Network(imagenet_pretrained=False).to(self.device).eval()

        with torch.no_grad():
            r_sg, r_s5, r_s4, r_s3 = res2net_net(self.dummy_input)
            g_sg, g_s5, g_s4, g_s3 = ghostnet_net(self.dummy_input)
            s_sg, s_s5, s_s4, s_s3 = shufflenet_net(self.dummy_input)

        expected_shape = (1, 1, 352, 352)
        # Verify Res2Net reference
        for name, tensor in [('S_g', r_sg), ('S_5', r_s5), ('S_4', r_s4), ('S_3', r_s3)]:
            self.assertEqual(tensor.shape, expected_shape, f"Res2Net {name} shape mismatch")

        # Verify GhostNet matches Res2Net
        self.assertEqual(g_sg.shape, r_sg.shape, "GhostNet S_g shape mismatch with Res2Net")
        self.assertEqual(g_s5.shape, r_s5.shape, "GhostNet S_5 shape mismatch with Res2Net")
        self.assertEqual(g_s4.shape, r_s4.shape, "GhostNet S_4 shape mismatch with Res2Net")
        self.assertEqual(g_s3.shape, r_s3.shape, "GhostNet S_3 shape mismatch with Res2Net")
        print("[PASS] GhostNet-SINet outputs match Res2Net-SINet (all 4 predictions are 1x1x352x352)")

        # Verify ShuffleNet matches Res2Net
        self.assertEqual(s_sg.shape, r_sg.shape, "ShuffleNet S_g shape mismatch with Res2Net")
        self.assertEqual(s_s5.shape, r_s5.shape, "ShuffleNet S_5 shape mismatch with Res2Net")
        self.assertEqual(s_s4.shape, r_s4.shape, "ShuffleNet S_4 shape mismatch with Res2Net")
        self.assertEqual(s_s3.shape, r_s3.shape, "ShuffleNet S_3 shape mismatch with Res2Net")
        print("[PASS] ShuffleNetV2-SINet outputs match Res2Net-SINet (all 4 predictions are 1x1x352x352)")


if __name__ == '__main__':
    unittest.main()
