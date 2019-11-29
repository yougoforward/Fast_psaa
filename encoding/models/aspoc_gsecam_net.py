from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['ASPOC_GSECAM_Net', 'get_aspoc_gsecamnet']


class ASPOC_GSECAM_Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ASPOC_GSECAM_Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = ASPOC_GSECAM_NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class ASPOC_GSECAM_NetHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(ASPOC_GSECAM_NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True)) if jpu else \
            nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                          norm_layer(inter_channels),
                          nn.ReLU(inplace=True))
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True)) if jpu else \
            nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                          norm_layer(inter_channels),
                          nn.ReLU(inplace=True))
        
        self.aa_aspp = aa_ASPP_Module(inter_channels, 256, atrous_rates, norm_layer, up_kwargs)

        self.sec = guided_SE_CAM_Module(inter_channels, 256, 512, norm_layer)


        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(512, num_classes)

    def forward(self, x):
        # aa_x = self.conv5a(x)
        aspp_feat = self.aa_aspp(x)
        # ss_x = self.conv5c(x)
        # sec_feat = self.sec(ss_x)
        # feat_sum = aspp_feat+sec_feat
        feat_sum = aspp_feat

        if self.se_loss:
            gap_feat = self.gap(feat_sum)
            gamma = self.fc(gap_feat)
            outputs = [self.conv8(F.relu_(feat_sum + feat_sum * gamma))]
            outputs.append(self.selayer(torch.squeeze(gap_feat)))
        else:
            outputs = [self.conv8(feat_sum)]
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



class SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels, norm_layer, scale=1):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
                context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)

        output = self.conv_bn_dropout(context)
        return output

class aa_ASPP_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(aa_ASPP_Module, self).__init__()
        self._up_kwargs = up_kwargs
        # out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.context = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
                                   norm_layer(out_channels),nn.ReLU(),
                                   SelfAttentionBlock(in_channels=out_channels, out_channels=out_channels, key_channels=out_channels//2, value_channels=out_channels,
                                    dropout=0, scale=2))

        # self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.sec = guided_SE_CAM_Module(in_channels, out_channels, out_channels, norm_layer)

        self.guided_se_cam = guided_SE_CAM_Module(out_channels, out_channels, out_channels, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        # feat4 = self.b4(x)
        oc = self.context(x)
        sec = self.sec(x)
        y = torch.cat((feat0, feat1, feat2, feat3, oc, sec), 1)
        out = self.guided_se_cam(y)
        return out


def get_aspoc_gsecamnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = ASPOC_GSECAM_Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model



class guided_CAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, query_dim, out_dim):
        super(guided_CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_dim = query_dim
        self.chanel_out = out_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, guide):
        """
            inputs :
                x=[x1,x2]
                x1 : input feature maps( B X C*5 X H X W)
                x2 : input deature maps (BxCxHxW)
            returns :
                out : output feature maps( B X C X H X W)
        """

        m_batchsize, C, height, width = x.size()

        proj_c_query = guide
        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_c_query.view(m_batchsize, self.query_dim, -1), proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out_c = torch.bmm(attention, x.view(m_batchsize, -1, width * height))
        out_c = out_c.view(m_batchsize, -1, height, width)
        return out_c


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


class guided_SE_CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, query_dim, out_dim, norm_layer):
        super(guided_SE_CAM_Module, self).__init__()

        self.project = nn.Sequential(
            nn.Conv2d(in_dim, query_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(query_dim), nn.ReLU(True),
        )

        self.guided_cam = guided_CAM_Module(in_dim, query_dim, query_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(query_dim), nn.ReLU(True),
        )

        self.se = SE_Module(in_dim, query_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.out = nn.Sequential(
            nn.Conv2d(query_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_dim), nn.ReLU(True),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        bottle = self.project(x)
        bottle = self.gamma*self.guided_cam(x, bottle)+bottle
        # bottle = self.fuse(bottle)
        se_x = self.se(x)
        se_bottle = self.relu(se_x * bottle + bottle)
        out = self.out(se_bottle)
        return out


