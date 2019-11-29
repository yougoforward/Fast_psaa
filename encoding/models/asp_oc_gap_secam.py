from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['ASP_OC_GAP_SECAMNet', 'get_asp_oc_gap_secamnet']


class ASP_OC_GAP_SECAMNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ASP_OC_GAP_SECAMNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = ASP_OC_GAP_SECAMNetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class ASP_OC_GAP_SECAMNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(ASP_OC_GAP_SECAMNetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
        #                             norm_layer(512),
        #                             nn.ReLU(inplace=True)) if jpu else \
        #     nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                   norm_layer(512),
        #                   nn.ReLU(inplace=True))
        # self.conv5as = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
        #                              norm_layer(512),
        #                              nn.ReLU(inplace=True)) if jpu else \
        #     nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                   norm_layer(512),
        #                   nn.ReLU(inplace=True))
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
        #                             norm_layer(512),
        #                             nn.ReLU(inplace=True)) if jpu else \
        #     nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                   norm_layer(512),
        #                   nn.ReLU(inplace=True))
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU(inplace=True))

        # self.sa = PAM_Module(inter_channels, inter_channels // 8, inter_channels)
        # self.sa = topk_PAM_Module(inter_channels, 256, inter_channels, 10)
        self.aa_aspp = aa_ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        # self.sec = SE_CAM_Module(inter_channels)
        self.sec = guided_SE_CAM_Module(in_channels, 256, 256, norm_layer)

        # self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels), nn.ReLU(True))
        self.conv52 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    norm_layer(256), nn.ReLU(True))
        # self.conv53 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels), nn.ReLU(True))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, out_channels, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Sigmoid())

        if self.se_loss:
            self.selayer = nn.Linear(256, out_channels)

    def forward(self, x):
        # ssa
        # feat1 = self.conv5a(x)
        # sa_feat = self.sa(feat1)
        # sa_conv = self.conv51(sa_feat)
        # aaspp
        # feat_as = self.conv5as(x)
        aspp_feat = self.aa_aspp(x)
        # aspp_conv = self.conv52(aspp_feat)
        # sec
        # feat2 = self.conv5c(x)
        sec_feat = self.sec(x)
        # sec_conv = self.conv53(sec_feat)
        # fuse
        # feat_sum = aspp_conv + sec_conv + sa_conv
        # outputs = self.conv8(feat_sum)
        feat_sum = aspp_feat+sec_feat
        # feat_sum = aspp_conv
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

class PyramidAttentionPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidAttentionPooling, self).__init__()
        out_channels = int(in_channels)
        # self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.pool2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.pool3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.pool4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))


        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = PAM_Module(in_channels, in_channels//4, out_channels)
        self.conv3 = PAM_Module(in_channels, in_channels//4, out_channels)
        self.conv4 = PAM_Module(in_channels, in_channels//4, out_channels)
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, _, h, w = x.size()
        d1 = x
        d2=self.pool2(d1)
        d3=self.pool3(d2)
        d4=self.pool4(d3)

        p2 = self.conv2(d2)
        p3 = self.conv3(d3)
        p4 = self.conv4(d4)

        u4 = F.upsample(p4, tuple(p3.size()[-2:]), **self._up_kwargs)
        u3 = F.upsample(p3 + u4, tuple(p2.size()[-2:]), **self._up_kwargs)
        u2 = F.upsample(p2 + u3, tuple(d1.size()[-2:]), **self._up_kwargs)

        out=d1+self.gamma*u2
        return out

class aa_ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(aa_ASPP_Module, self).__init__()
        self._up_kwargs = up_kwargs
        out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.pap = PyramidAttentionPooling(out_channels, norm_layer, up_kwargs)

        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.guided_se_cam = guided_SE_CAM_Module(5 * out_channels, out_channels, out_channels, norm_layer)

        self.guided_se_cam0 = guided_SE_CAM_Module(out_channels, out_channels, out_channels, norm_layer)
        self.guided_se_cam1 = guided_SE_CAM_Module(out_channels, out_channels, out_channels, norm_layer)
        self.guided_se_cam2 = guided_SE_CAM_Module(out_channels, out_channels, out_channels, norm_layer)
        self.guided_se_cam3 = guided_SE_CAM_Module(out_channels, out_channels, out_channels, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        # feat0 = self.b0(x)
        # feat1 = self.b1(x)
        # feat2 = self.b2(x)
        # feat3 = self.b3(x)

        feat0 = self.guided_se_cam0(self.pap(self.b0(x)))
        feat1 = self.guided_se_cam1(self.b1(x))
        feat2 = self.guided_se_cam2(self.b2(x))
        feat3 = self.guided_se_cam3(self.b3(x))

        feat4 = self.b4(x)

        # y = torch.cat((feat0, feat1, feat2, feat3), 1)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        out = self.guided_se_cam(y)
        return out


def get_asp_oc_gap_secamnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = ASP_OC_GAP_SECAMNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, key_dim, out_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value.view(m_batchsize, -1, width * height), attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + proj_value
        return out



class topk_PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, key_dim, out_dim, topk=10):
        super(topk_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.topk = topk
        self.key_channels = key_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Mask_Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        proj_value = self.value_conv(x)
        proj_value = proj_value.view(m_batchsize, -1, width * height)

        # attention mask selection
        val, idx = torch.topk(energy, height * width // self.topk, dim=2, largest=True, sorted=False)
        at_sparse = torch.zeros_like(energy).cuda()
        attention_mask = at_sparse.scatter_(2, idx, 1.0)

        attention = self.softmax([energy, attention_mask])

        # for inference with batch 1
        # energy_sp = topk2sparse(idx, val)
        # attention_sp = sparse_softmax(energy_sp)
        # out = torch.sparse.mm(attention_sp,proj_value.permute(0,2,1)).permute(0,2,1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class guided_CAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, query_dim, out_dim):
        super(guided_CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_dim = query_dim
        self.chanel_out = out_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv_c = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=query_dim, kernel_size=1, bias=False), nn.BatchNorm2d(query_dim),
            nn.ReLU(), nn.Dropout2d(0.1))

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
        out_c = self.gamma * out_c + proj_c_query
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
        self.guided_cam = guided_CAM_Module(in_dim, query_dim, query_dim)
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, query_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(query_dim), nn.ReLU(True),
            nn.Dropout2d(0.1)
        )
        self.se = SE_Module(in_dim, query_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(query_dim*2, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_dim), nn.ReLU(True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gcam = self.guided_cam(x)

        bottle = self.project(x)
        se_x = self.se(x)
        se_bottle = se_x * bottle + bottle
        out = torch.cat([gcam, se_bottle], dim=1)
        out =self.fuse(out)
        return out


class SE_CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(SE_CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.se = SE_Module(in_dim, in_dim)

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

        se_x = self.se(x)
        se_out = se_x * x

        out = se_out + self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

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

        out = self.gamma * out + x
        return out

