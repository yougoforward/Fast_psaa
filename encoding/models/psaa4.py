from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet
from ..nn import dcn

__all__ = ['psaa4Net', 'get_psaa4net']


class psaa4Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa4Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa4NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        x = list(self.head(c4, c2, c1))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class psaa4NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(6, 12, 18)):
        super(psaa4NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_psaa4 = psaa4_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256+inter_channels, out_channels, 1))
        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x, c2, c1):
        feat_sum, gap_feat = self.aa_psaa4(x, c2, c1)
        outputs = [self.conv8(feat_sum)]
        if self.se_loss:
            outputs.append(self.selayer(torch.squeeze(gap_feat)))

        return tuple(outputs)


def psaa4Conv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class psaa4Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa4Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

        self.out_chs = out_channels

    def forward(self, x):
        bs, _, h, w = x.size()
        pool = self.gap(x)

        # return F.interpolate(pool, (h, w), **self._up_kwargs)
        # return pool.repeat(1,1,h,w)
        return pool.expand(bs, self.out_chs, h, w)


class psaa4_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa4_Module, self).__init__()
        # out_channels = in_channels // 4
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa4Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa4Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa4Conv(in_channels, out_channels, rate3, norm_layer)
        # self.b4 = psaa4Conv(in_channels, out_channels, rate4, norm_layer)
        # self.b4 = psaa4Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self._up_kwargs = up_kwargs
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+4*out_channels, out_channels, 1, padding=0, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, 4, 1, bias=True))        
        self.project = nn.Sequential(nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))


        # self.gap4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                     nn.Conv2d(512, 256, 1, bias=False),
        #                     norm_layer(256),
        #                     nn.ReLU(True))
        self.se4 = nn.Sequential(
                            nn.Conv2d(512, 256, 1, bias=True),
                            nn.Sigmoid())

        # self.gap8 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                     nn.Conv2d(512, 512, 1, bias=False),
        #                     norm_layer(512),
        #                     nn.ReLU(True))
        self.se8 = nn.Sequential(
                            nn.Conv2d(512, 512, 1, bias=True),
                            nn.Sigmoid())

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, out_channels, 1, bias=False),
                            norm_layer(out_channels),
                            nn.ReLU(True))
        self.se16 = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())

        self.skip8 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=32,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(32),
                      nn.ReLU(True))
        self.skip4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=16,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(16),
                      nn.ReLU(True))
        self.project8to4 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=256,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(256),
                      nn.ReLU(True))
        self.project16to8 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(512),
                      nn.ReLU(True))

        self.up16to8 = nn.Sequential(
            dcn.DFConv2d(
                512+32,
                512,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), norm_layer(512), nn.ReLU(inplace=True),
            dcn.DFConv2d(
                512,
                512,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), norm_layer(512), nn.ReLU(inplace=True)
        )
        self.up8to4 = nn.Sequential(
            dcn.DFConv2d(
                256+16,
                256,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), norm_layer(256), nn.ReLU(inplace=True),
            dcn.DFConv2d(
                256,
                256,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), norm_layer(256), nn.ReLU(inplace=True)
        )
    def forward(self, x, c2, c1):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        # feat4 = self.b4(x)
        n, c, h, w = feat0.size()

        # psaa
        y1 = torch.cat((feat0, feat1, feat2, feat3), 1)
        psaa_feat = self.psaa_conv(torch.cat([x, y1], dim=1))
        psaa_att = torch.sigmoid(psaa_feat)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
                        psaa_att_list[3] * feat3), 1)
        out = self.project(y2)

        #gp
        gp = self.gap(x)
        se16 = self.se16(gp)
        out = torch.cat([out+se16*out, gp.expand(n, c, h, w)], dim=1)
        #upsampling
        _, _, h8, w8 = c2.size()
        _, cc, h4, w4 = c1.size()

        #16 to 8
        x8 = self.skip8(c2)
        out = self.project16to8(out)
        out = F.interpolate(out, (h8, w8), **self._up_kwargs)
        x8 = torch.cat([x8, out], dim=1)
        out = self.up16to8(x8)
        se8 = self.se8(gp)
        out = torch.cat([out+se8*out, gp.expand(n, c, h8, w8)], dim=1)

        #8 to 4
        x4 = self.skip4(c1)
        out = self.project8to4(out)
        out = F.interpolate(out, (h4, w4), **self._up_kwargs)
        x4 = torch.cat([x4, out], dim=1)
        out = self.up8to4(x4)
        
        #gp
        se4 = self.se4(gp)
        out = torch.cat([out+se4*out, gp.expand(n, c, h4, w4)], dim=1)
        return out, gp

def get_psaa4net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa4Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


