import torch.nn as nn
import torch
import torch.nn.functional as F
from network.vgg import VGG16

class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x

class TextNet(nn.Module):

    def __init__(self, backbone='vgg', output_channel=7, is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.output_channel = output_channel

        if backbone == 'vgg':
            self.backbone = VGG16(pretrain=self.is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512 + 256, 128)
            self.merge3 = Upsample(256 + 128, 64)
            self.merge2 = Upsample(128 + 64, 32)
            self.merge1 = Upsample(64 + 32, 16)
            self.predict = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, self.output_channel, kernel_size=1, stride=1, padding=0)
            )
            # ── Embedding head (parallel to predict, reads same up1) ──
            self.embedding_head = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)  # 8 = EMBEDDING_DIM
            )
        elif backbone == 'resnet':
            pass

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)
        prediction = self.predict(up1)
        embedding = self.embedding_head(up1)   # (B, 8, H, W)

        return prediction, embedding

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'], strict=False)

if __name__ == '__main__':
    import torch

    input = torch.randn((4, 3, 512, 512))
    net = TextNet().cuda()
    output = net(input.cuda())
    print(output.size())



### edited by subhra added resnet backbone support
# Note: The ResNet backbone implementation is not provided in the original snippet.
# If you need a ResNet backbone, you can implement it similarly to the VGG backbone
# or use a pre-existing implementation from libraries like torchvision.
# If you want to use a ResNet backbone, you would typically import it from torchvision.models
# and adapt the forward method accordingly.



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# from network.vgg import VGG16            # keep your existing VGG helper


# # ───────────────────────────────── Upsampling block ─────────────────────────────
# class Upsample(nn.Module):
#     """Concat → 1×1 conv → ReLU → 3×3 conv → ReLU → deconv(×2)."""

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1x1 = nn.Conv2d(in_channels, in_channels,
#                                  kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(in_channels, out_channels,
#                                  kernel_size=3, stride=1, padding=1)
#         self.deconv  = nn.ConvTranspose2d(out_channels, out_channels,
#                                           kernel_size=4, stride=2, padding=1)

#     def forward(self, upsampled, shortcut):
#         x = torch.cat([upsampled, shortcut], dim=1)
#         x = F.relu(self.conv1x1(x))
#         x = F.relu(self.conv3x3(x))
#         return self.deconv(x)


# # ─────────────────────────────── ResNet feature trunk ───────────────────────────
# class ResNetBackbone(nn.Module):
#     """
#     Returns C1–C5 feature maps at strides 2, 4, 8, 16, 32 w.r.t. input.
#         C1 : 64   ch,  H/2
#         C2 : 256  ch,  H/4
#         C3 : 512  ch,  H/8
#         C4 : 1024 ch, H/16
#         C5 : 2048 ch, H/32
#     """

#     def __init__(self, pretrained: bool = True, resnet_type: str = "resnet50"):
#         super().__init__()
#         resnet = getattr(models, resnet_type)(pretrained=pretrained)

#         self.conv1  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1   # 256  ch
#         self.layer2 = resnet.layer2   # 512  ch
#         self.layer3 = resnet.layer3   # 1024 ch
#         self.layer4 = resnet.layer4   # 2048 ch

#     def forward(self, x):
#         C1 = self.conv1(x)            # /2
#         x  = self.maxpool(C1)         # /4
#         C2 = self.layer1(x)           # /4
#         C3 = self.layer2(C2)          # /8
#         C4 = self.layer3(C3)          # /16
#         C5 = self.layer4(C4)          # /32
#         return C1, C2, C3, C4, C5


# # ───────────────────────────────────── TextNet ──────────────────────────────────
# class TextNet(nn.Module):
#     """
#     Text detection network used in CRAFT-style pipelines.
#     Supports 'vgg' (original paper) and 'resnet' backbones.
#     """

#     def __init__(self, backbone: str = "vgg", output_channel: int = 7,
#                  is_training: bool = True):
#         super().__init__()

#         self.output_channel = output_channel
#         self.is_training    = is_training

#         # ────────────── VGG16 pathway (unchanged from your code) ───────────────
#         if backbone == "vgg":
#             self.backbone = VGG16(pretrain=self.is_training)

#             self.deconv5 = nn.ConvTranspose2d(512, 256,
#                                               kernel_size=4, stride=2, padding=1)
#             self.merge4  = Upsample(512 + 256, 128)   # C4 + up5
#             self.merge3  = Upsample(256 + 128, 64)    # C3 + up4
#             self.merge2  = Upsample(128 + 64,  32)    # C2 + up3
#             self.merge1  = Upsample(64  + 32,  16)    # C1 + up2

#         # ────────────── ResNet-50 pathway (new) ────────────────────────────────
#         elif backbone == "resnet":
#             self.backbone = ResNetBackbone(pretrained=self.is_training,
#                                             resnet_type="resnet50")

#             self.deconv5 = nn.ConvTranspose2d(2048, 256,
#                                               kernel_size=4, stride=2, padding=1)
#             self.merge4  = Upsample(1024 + 256, 128)  # C4 + up5
#             self.merge3  = Upsample(512  + 128, 64)   # C3 + up4
#             self.merge2  = Upsample(256  + 64,  32)   # C2 + up3
#             self.merge1  = Upsample(64   + 32,  16)   # C1 + up2
#         else:
#             raise ValueError("backbone must be 'vgg' or 'resnet'")

#         # prediction head (shared)
#         self.predict = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(16, self.output_channel, kernel_size=1, stride=1, padding=0)
#         )

#     # ───────────────────────────── forward pass ────────────────────────────────
#     def forward(self, x):
#         C1, C2, C3, C4, C5 = self.backbone(x)

#         up5 = F.relu(self.deconv5(C5))
#         up4 = F.relu(self.merge4(up5, C4))
#         up3 = F.relu(self.merge3(up4, C3))
#         up2 = F.relu(self.merge2(up3, C2))
#         up1 = self.merge1(up2, C1)        # final upsample doesn’t need extra ReLU
#         return self.predict(up1)

#     # ─────────────────────────── utility: load saved ckpt ──────────────────────
#     def load_model(self, model_path: str):
#         print(f"Loading weights from {model_path}")
#         state_dict = torch.load(model_path, map_location="cpu")
#         self.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)


# # ──────────────────────────────────── test ──────────────────────────────────────
# if __name__ == "__main__":
#     inp = torch.randn(4, 3, 512, 512).cuda()

#     net_vgg = TextNet(backbone="vgg").cuda()
#     net_res = TextNet(backbone="resnet").cuda()

#     print("VGG   output:", net_vgg(inp).shape)   # ➜ (4, 7, 512, 512)
#     print("ResNet output:", net_res(inp).shape)  # ➜ (4, 7, 512, 512)
