import torch
import torch.nn as nn
from models.network_swinfusion import SwinFusion

from utils.tensor_ops import cus_sample, upsample_add
from backbone.VGG import (
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
from module.MyModules import (
    EDFM,
    IDEM,
    FDM,
)

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class DEFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DEFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.t_trans16 = IDEM(512, 64)
        self.t_trans8 = IDEM(512, 64)
        self.t_trans4 = IDEM(256, 64)
        self.t_trans2 = IDEM(128,32)
        self.t_trans1 = IDEM(64,64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)


        self.selfdc_16 = EDFM(64, 64)
        self.selfdc_8 = EDFM(64, 64)
        self.selfdc_4 = EDFM(64, 64)
        self.selfdc_2 = EDFM(32,32)
        self.selfdc_1 = EDFM(32,32)





        self.fdm = FDM()

        # ==========
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v2 = nn.Sequential(
            nn.Conv2d(64+32, 60, 3, padding=1, dilation=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v3 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v4 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v5 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.model = SwinFusion(window_size=8, embed_dim=60)

    def fusion_(self, shared1, shared2, shared3, shared4, shared5):
        shared5 = self.up5(shared5)#64*80*64
        shared4 = torch.cat((shared4, shared5), dim=1)#128*80*64
        shared4 = self.v5(shared4)#64*80*64

        shared4 = self.up4(shared4)  # 64*160*120
        shared4 = torch.cat((shared4, shared3), dim=1)  # 128*160*128
        shared3 = self.v4(shared4)  # 64*160*128
        shared3 = self.up3(shared3)  # 64*320*256
        shared3 = torch.cat((shared3, shared2), dim=1)  # 96*320*256
        shared2 = self.v3(shared3)  # 64*320*256
        shared2 = self.up2(shared2)  # 64*640*512
        shared2 = torch.cat((shared2, shared1), dim=1)  # 96*640*512
        shared1 = self.v2(shared2)  # 60*640*512
        # shared1 = self.up1(shared1)  # 60*128*128->60*256*256
        return shared1

    def forward(self, RGB,T):
        in_data = RGB
        in_depth = T
        in_data_1 = self.encoder1(in_data)


        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)


        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)


        in_data_1_aux = self.t_trans1(in_data_1,in_data_1_d)
        in_data_2_aux = self.t_trans2(in_data_2,in_data_2_d)
        in_data_4_aux = self.t_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.t_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.t_trans16(in_data_16, in_data_16_d)

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024

        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        out_data_8 = self.upconv8(out_data_8)  # 512

        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        out_data_4 = self.upconv4(out_data_4)  # 256

        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        out_data_2 = self.upconv2(out_data_2)  # 64

        out_data_1 = self.upsample_add(self.selfdc_2(out_data_2,in_data_2_aux),in_data_1)
        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.fdm(out_data_1,out_data_2,out_data_4,out_data_8,out_data_16)
        z = self.fusion_(out_data_1,out_data_2,out_data_4,out_data_8,out_data_16)
        z = self.model(z)
        return [out_data,z]



def fusion_model():
    model = DEFNet()
    return model

if __name__ == "__main__":
    model = DEFNet()
    x = torch.randn(2,3,640,480)
    depth = torch.randn(2,3,640,480)
    fuse = model(x,depth)
    print(fuse.shape)

