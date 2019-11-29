from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['aspoc_psaa_gpNet', 'get_aspoc_psaa_gpnet']


class aspoc_psaa_gpNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(aspoc_psaa_gpNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = aspoc_psaa_gpNetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class aspoc_psaa_gpNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(aspoc_psaa_gpNetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aspoc = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(inter_channels),nn.ReLU(True),
            ASPOC_Module(inter_channels, inter_channels//2, norm_layer, scale=2)
        )

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1)) # gp

    def forward(self, x):

        feat_sum = self.aspoc(x)
        outputs = [self.conv8(feat_sum)]

        return tuple(outputs)


def aspoc_psaa_gpConv(in_channels, out_channels, atrous_rate, norm_layer):
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


class gap_Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(gap_Pooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

        self.out_chs = out_channels

    def forward(self, x):
        bs, _, h, w = x.size()
        pool = self.gap(x)
        return pool.expand(bs, self.out_chs, h, w)



def get_aspoc_psaa_gpnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = aspoc_psaa_gpNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class SelfAttentionModule(nn.Module):
    """The basic implementation for self-attention block/non-local block
    Parameters:
        in_dim       : the dimension of the input feature map
        key_dim      : the dimension after the key/query transform
        value_dim    : the dimension after the value transform
        scale        : choose the scale to downsample the input feature maps (save memory cost)
    """

    def __init__(self, in_dim, out_dim, key_dim, value_dim, norm_layer, scale=2):
        super(SelfAttentionModule, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.func_key = nn.Sequential(nn.Conv2d(in_channels=self.in_dim, out_channels=self.key_dim,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      norm_layer(self.key_dim), nn.ReLU(True))
        self.func_query = self.func_key
        self.func_value = nn.Conv2d(in_channels=self.in_dim, out_channels=self.value_dim,
                                    kernel_size=1, stride=1, padding=0)
        self.weights = nn.Conv2d(in_channels=self.value_dim, out_channels=self.out_dim,
                                 kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.weights.weight, 0)
        nn.init.constant_(self.weights.bias, 0)

        self.refine = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                    norm_layer(out_dim), nn.ReLU(True))

    def forward(self, x):
        batch, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.func_value(x).view(batch, self.value_dim, -1)  # bottom
        value = value.permute(0, 2, 1)
        query = self.func_query(x).view(batch, self.key_dim, -1)  # top
        query = query.permute(0, 2, 1)
        key = self.func_key(x).view(batch, self.key_dim, -1)  # mid

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch, self.value_dim, *x.size()[2:])
        context = self.weights(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        output = self.refine(context)
        return output


class ASPOC_Module(nn.Module):
    """ASPP with OC module: aspp + oc context"""

    def __init__(self, in_dim, out_dim, norm_layer, scale):
        super(ASPOC_Module, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                                         norm_layer(out_dim), nn.ReLU(True),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, norm_layer=norm_layer, scale=scale))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        norm_layer(out_dim), nn.ReLU(True))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        norm_layer(out_dim), nn.ReLU(True))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=24, dilation=24, bias=False),
                                        norm_layer(out_dim), nn.ReLU(True))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=36, dilation=36, bias=False),
                                        norm_layer(out_dim), nn.ReLU(True))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 5, 2*out_dim, kernel_size=1, padding=0, bias=False),
                                       norm_layer(out_dim), nn.ReLU(True),
                                       )


        self.psaa_conv = nn.Sequential(nn.Conv2d(in_dim+5*out_dim, out_dim, 1, padding=0, bias=False),
                                    norm_layer(out_dim),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_dim, 5, 1, bias=True))
        self.project = nn.Sequential(nn.Conv2d(in_channels=5*out_dim, out_channels=2*out_dim,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(2*out_dim),
                      nn.ReLU(True))
        self.gap = gap_Pooling(in_dim, 2*out_dim, norm_layer)

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        # psaa
        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        psaa_feat = self.psaa_conv(torch.cat([x,y1], dim=1))
        psaa_att = torch.sigmoid(psaa_feat)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0]*feat0, psaa_att_list[1]*feat1, psaa_att_list[2]*feat2, psaa_att_list[3]*feat3, psaa_att_list[4]*feat4), 1)
        out = self.project(y2)
        gp = self.gap(x)
        out = torch.cat([out, gp], dim=1)
        return out