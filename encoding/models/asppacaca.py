from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['asppacaca', 'get_asppacaca']


class asppacaca(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(asppacaca, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = asppacaca_head(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class asppacaca_head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(asppacaca_head, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4
        aspp_out_channels= 256
        self.conv5as = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(512),
                                     nn.ReLU(inplace=True)) if jpu else \
            nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                          norm_layer(512),
                          nn.ReLU(inplace=True))

        self.aa_aspp = aa_ASPP_Module(inter_channels, atrous_rates, norm_layer, up_kwargs)


        self.conv52 = nn.Sequential(nn.Conv2d(aspp_out_channels, aspp_out_channels, 3, padding=1, bias=False),
                                    norm_layer(256), nn.ReLU(True))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(aspp_out_channels, out_channels, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(aspp_out_channels, aspp_out_channels, 1),
            nn.Sigmoid())

        if self.se_loss:
            self.selayer = nn.Linear(aspp_out_channels, out_channels)

    def forward(self, x):
        # aaspp
        feat_as = self.conv5as(x)
        aspp_feat = self.aa_aspp(feat_as)
        aspp_conv = self.conv52(aspp_feat)

        if self.se_loss:
            gap_feat = self.gap(aspp_conv)
            gamma = self.fc(gap_feat)
            outputs = [self.conv8(F.relu_(aspp_conv + aspp_conv * gamma))]
            outputs.append(self.selayer(torch.squeeze(gap_feat)))
        else:
            outputs = [self.conv8(aspp_conv)]

        return tuple(outputs)


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), **self._up_kwargs)


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)


class aa_ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(aa_ASPP_Module, self).__init__()
        out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        # self.b0 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        # )
        self.b1 = nn.Sequential(ASPPConv(in_channels, out_channels, rate1, norm_layer))
        self.b2 = nn.Sequential(ASPPConv(in_channels, out_channels, rate2, norm_layer))
        self.b3 = nn.Sequential(ASPPConv(in_channels, out_channels, rate3, norm_layer))

        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            CA_Module(out_channels))
        self.b5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            PA_Module(out_channels, out_channels, out_channels//2, out_channels, scale=2, norm_layer=norm_layer))

        # self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.guided_ca = guided_CA_Module(5 * out_channels, out_channels, out_channels, norm_layer)

    def forward(self, x):
        # feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat5 = self.b5(x)
        y = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        out = self.guided_ca(y)
        return out


def get_asppacaca(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = asppacaca(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

class PA_Module(nn.Module):
    """The basic implementation for self-attention block/non-local block
    Parameters:
        in_dim       : the dimension of the input feature map
        key_dim      : the dimension after the key/query transform
        value_dim    : the dimension after the value transform
        scale        : choose the scale to downsample the input feature maps (save memory cost)
    """

    def __init__(self, in_dim, out_dim, key_dim, value_dim, scale=2, norm_layer= nn.BatchNorm2d):
        super(PA_Module, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.func_key = nn.Sequential(nn.Conv2d(in_channels=self.in_dim, out_channels=self.key_dim,
                                                kernel_size=1, stride=1, padding=0),
                                      norm_layer(self.key_dim), nn.ReLU(False))
        self.func_query = self.func_key
        self.func_value = nn.Conv2d(in_channels=self.in_dim, out_channels=self.value_dim,
                                    kernel_size=1, stride=1, padding=0)
        self.weights = nn.Conv2d(in_channels=self.value_dim, out_channels=self.out_dim,
                                 kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.weights.weight, 0)
        nn.init.constant_(self.weights.bias, 0)

        self.refine = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0),
                                    norm_layer(out_dim), nn.ReLU(False))


    def forward(self, x):
        batch, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.func_value(x).view(batch, self.value_dim, -1)  # bottom
        value = value.permute(0, 2, 1)
        query = self.func_query(x).view(batch, self.key_dim, -1)  # top
        query = query.permute(0, 2, 1)
        key = self.func_key(x).view(batch, self.key_dim, -1)  # mid

        sim_map = torch.bmm(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.bmm(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch, self.value_dim, *x.size()[2:])
        context = self.weights(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        output = self.refine(context)
        return output


class guided_CA_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, query_dim, out_dim, norm_layer):
        super(guided_CA_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_dim = query_dim
        self.chanel_out = out_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv_c = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=query_dim, kernel_size=1, bias=False), norm_layer(query_dim),
            nn.ReLU(), nn.Dropout2d(0.1))

        self.se = SE_Module(query_dim*2, out_dim)

    def forward(self, x):
        """
            inputs :
                x=[x1,x2]
                x1 : input feature maps( B X C*5 X H X W)
                x2 : input deature maps (BxCxHxW)
            returns :
                out : output feature maps( B X C X H X W)
        """

        m_batchsize, C, height, width = x.size()
        proj_c_query = self.query_conv_c(x)

        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_c_query.view(m_batchsize, self.query_dim, -1), proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out_c = torch.bmm(attention, x.view(m_batchsize, -1, width * height))
        out_c = out_c.view(m_batchsize, -1, height, width)

        # out_c = self.gamma * out_c + proj_c_query
        se_x = self.se(torch.cat([proj_c_query, out_c], dim=1))

        out = se_x * proj_c_query + (1 - se_x) * out_c
        return out


class SE_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(SE_Module, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(in_dim, in_dim // 16, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_dim // 16, out_dim, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        out = self.se(x)
        return out

class CA_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CA_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.se = SE_Module(in_dim*2, in_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        se_x = self.se(torch.cat([x, out], dim=1))

        out = se_x * x + (1-se_x) * out
        return out




