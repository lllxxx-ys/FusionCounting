# Cross-modal Spatio-Channel Attention (CSCA) block

import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from models.network_swinfusion import SwinFusion

class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        c1 = int(64 * ratio)# 38, ratio=0.6
        c2 = int(128 * ratio)# 76
        c3 = int(256 * ratio)# 153
        c4 = int(512 * ratio)# 307

        self.block1_depth = Block([c1, c1, 'M'], in_channels=3, L=4, first_block=True, D_in_channels=True,dropout_rate=0.1)
        self.block1 = Block([c1, c1, 'M'], in_channels=3, L=4, first_block=True, D_in_channels=False,dropout_rate=0.1)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1, L=3,dropout_rate=0.25)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2, L=2,dropout_rate=0.35)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3, L=1,dropout_rate=0.45)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4, L=1,dropout_rate=0.)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()


        #==========
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v2 = nn.Sequential(
            nn.Conv2d(128+38, 60, 3, padding=1, dilation=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v3 = nn.Sequential(
            nn.Conv2d(76+256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.v4 = nn.Sequential(
            nn.Conv2d(153 + 307, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v5 = nn.Sequential(
            nn.Conv2d(307 + 307, 307, 3, padding=1, dilation=1),
            nn.BatchNorm2d(307),
            nn.ReLU(inplace=True)
        )

        self.model = SwinFusion(window_size=8, embed_dim=60)

    def fusion_(self,shared1,shared2,shared3,shared4,shared5):

        shared4=torch.cat((shared4, shared5), dim=1)
        shared4 = self.v5(shared4)

        shared4=self.up4(shared4)#307*16*16->307*32*32
        shared4 = torch.cat((shared4, shared3), dim=1)#307*32*32+153*32*32=460*32*32
        shared3=self.v4(shared4)#460*32*32->256*32*32
        shared3 = self.up3(shared3)#256*32*32->256*64*64
        shared3 = torch.cat((shared3, shared2), dim=1)#256*64*64+76*64*64
        shared2 = self.v3(shared3)#332*64*64->128*64*64
        shared2 = self.up2(shared2)#128*64*64->128*128*128
        shared2 = torch.cat((shared2, shared1), dim=1)#128*128*128*+38*128*128
        shared1 = self.v2(shared2)#166*128*128->60*128*128
        shared1=self.up1(shared1)#60*128*128->60*256*256
        return shared1

    def forward(self, RGBT, dataset):
        RGB = RGBT[0]
        T = RGBT[1]

        if dataset == 'ShanghaiTechRGBD':
            RGB, T, shared1 = self.block1_depth(RGB, T)
        else:
            RGB, T, shared1 = self.block1(RGB, T)
        RGB, T, shared2 = self.block2(RGB, T)
        RGB, T, shared3 = self.block3(RGB, T)
        RGB, T, shared4 = self.block4(RGB, T)
        _, _, shared5= self.block5(RGB, T)
        x = shared5
        #=============
        z=self.fusion_(shared1,shared2,shared3,shared4,shared5)
        z=self.model(z)

        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)

        return [torch.abs(x),z]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels, L, first_block=False, dilation_rate=1, D_in_channels=False,dropout_rate=0.):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate
        self.L = L
        self.dropout_rate=dropout_rate

        if first_block:
            if D_in_channels:
                t_in_channels = 1
            else:
                t_in_channels = in_channels
        else:
            t_in_channels = in_channels

        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=t_in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

        channels = cfg[0]
        self.out_channels = channels//2

        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)

        self.RGB_key = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(p=self.dropout_rate),  # 添加 Dropout
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_query = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(p=self.dropout_rate),  # 添加 Dropout
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.RGB_W = nn.Conv2d(in_channels=self.out_channels, out_channels=channels,
                           kernel_size=1, stride=1, padding=0)

        self.T_key = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(p=self.dropout_rate),  # 添加 Dropout
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.T_query = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(p=self.dropout_rate),  # 添加 Dropout
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.T_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.T_W = nn.Conv2d(in_channels=self.out_channels, out_channels=channels,
                           kernel_size=1, stride=1, padding=0)

        self.gate_RGB = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.gate_T = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, RGB, T):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)

        new_RGB, new_T, new_shared = self.fuse(RGB, T)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)

        # SCA Block
        adapt_channels = 2 ** self.L * self.out_channels
        batch_size = RGB_m.size(0)
        rgb_query = self.RGB_query(RGB_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        rgb_key = self.RGB_key(RGB_m).view(batch_size, adapt_channels, -1)
        rgb_value = self.RGB_value(RGB_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        batch_size = T_m.size(0)
        T_query = self.T_query(T_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        T_key = self.T_key(T_m).view(batch_size, adapt_channels, -1)
        T_value = self.T_value(T_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        RGB_sim_map = torch.matmul(T_query, rgb_key)
        RGB_sim_map = (adapt_channels ** -.5) * RGB_sim_map
        RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        RGB_context = torch.matmul(RGB_sim_map, rgb_value)
        RGB_context = RGB_context.permute(0, 2, 1).contiguous()
        RGB_context = RGB_context.view(batch_size, self.out_channels,  *RGB_m.size()[2:])
        RGB_context = self.RGB_W(RGB_context)

        T_sim_map = torch.matmul(rgb_query, T_key)
        T_sim_map = (adapt_channels ** -.5) * T_sim_map
        T_sim_map = F.softmax(T_sim_map, dim=-1)
        T_context = torch.matmul(T_sim_map, T_value)
        T_context = T_context.permute(0, 2, 1).contiguous()
        T_context = T_context.view(batch_size, self.out_channels, *T_m.size()[2:])
        T_context = self.T_W(T_context)


        # CFA Block
        cat_fea = torch.cat([T_context, RGB_context], dim=1)
        attention_vector_RGB = self.gate_RGB(cat_fea)
        attention_vector_T = self.gate_T(cat_fea)

        attention_vector = torch.cat([attention_vector_RGB, attention_vector_T], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_RGB, attention_vector_T = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        new_shared = RGB * attention_vector_RGB + T * attention_vector_T

        new_RGB = (RGB + new_shared) / 2
        new_T = (T + new_shared) / 2

        new_RGB = self.relu1(new_RGB)
        new_T = self.relu2(new_T)
        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion


def fusion_model():
    model = FusionModel()
    return model


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = FusionModel()
    # print(model)
    # print(height, width, model.flops() / 1e9)
    #
    x = torch.randn((2, 3, 255,255))
    z = model([x,x],'ss')
    print(z.shape)