from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['pgfNet', 'get_pgfnet']

class pgfNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(pgfNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = pgfNetHead(2048, nclass, se_loss, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            x.append(auxout)

        return tuple(x)


class pgfNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss, norm_layer):
        super(pgfNetHead, self).__init__()
        inter_channels = in_channels // 4

        self.pgf_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.pgf = PyramidGuidedFusion(inter_channels, se_loss, norm_layer)

        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))
        self.se_loss = se_loss
        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):
        x = self.pgf_conv(x)
        x = self.pgf(x)
        # x=[x]
        outputs = [self.block(x[0])]

        if self.se_loss:
            outputs.append(self.selayer(torch.squeeze(x[1])))
        return tuple(outputs)


class GuidedFusion(nn.Module):
    """
    exploit self-attentin for  adjacent scale fusion
    """
    def __init__(self, in_channels, query_dim, norm_layer):
        super(GuidedFusion, self).__init__()
        self.key_channels = query_dim
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=query_dim,
                      kernel_size=1, stride=1, padding=0)
        # self.query_conv =  nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=self.key_channels,
        #               kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(self.key_channels),
        #     nn.ReLU(True)
        # )

        self.value_conv =  nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(in_channels)
        )
        self.key_conv = self.query_conv

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.relu=nn.ReLU(True)

    def forward(self, low_level, high_level):

        m_batchsize, C, hl, wl = low_level.size()
        m_batchsize, C, hh, wh = high_level.size()

        # query = low_level.view(m_batchsize, C, hl * wl).permute(0, 2, 1)  # m, hl*wl, c
        # key = high_level.view(m_batchsize, C, hh * wh)  # m, c, hh*wh

        query = self.query_conv(low_level).view(m_batchsize, -1, hl * wl).permute(0, 2, 1) # m, hl*wl, c
        key = self.key_conv(high_level).view(m_batchsize, -1, hh * wh) # m, c, hh*wh

        value = self.value_conv(high_level)

        energy = torch.bmm(query, key)        # C, hl*wl,hh*wh

        energy = (self.key_channels ** -.5) * energy

        attention = self.softmax(energy)
        out = torch.bmm(value.view(m_batchsize, C, hh*wh), attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, hl, wl)

        out = self.relu(self.gamma * out + low_level)
        return out


class PyramidGuidedFusion(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, se_loss, norm_layer):
        super(PyramidGuidedFusion, self).__init__()

        self.pool2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.pool3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.pool4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))

        self.gf2 = GuidedFusion(in_channels, 128, norm_layer)
        self.gf3 = GuidedFusion(in_channels, 128, norm_layer)
        self.gf4 = GuidedFusion(in_channels, 128, norm_layer)

        self.se_loss = se_loss
        if self.se_loss:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.se = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1,
                                              bias=True),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        _, _, h, w = x.size()
        d1 = x
        d2=self.pool2(d1)
        d3=self.pool3(d2)
        d4=self.pool4(d3)
        if self.se_loss:
            gap_feat = self.gap(d4)
            gamma = self.se(gap_feat)
            d4 = F.relu(d4 + d4 * gamma)

        u3 = self.gf4(d3, d4)
        u2 = self.gf3(d2, u3)
        # u2=d2
        u1 = self.gf2(d1, u2)
        # u1=d1
        outputs= [u1]
        if self.se_loss:
            outputs.append(gap_feat)
        return tuple(outputs)




def get_pgfnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = pgfNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
