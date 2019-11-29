###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead

class PSP2(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP2, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSP2Head(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = upsample(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class PSP2Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSP2Head, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_psp2(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = PSP2(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp2('ade20k', 'resnet50', pretrained, root=root, **kwargs)

class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels/4)
        self.pam1 = PAM_Module(in_channels, out_channels, 1, norm_layer)
        self.pam2 = PAM_Module(in_channels, out_channels, 2, norm_layer)
        self.pam3 = PAM_Module(in_channels, out_channels, 3, norm_layer)
        self.pam4 = PAM_Module(in_channels, out_channels, 6, norm_layer)

    def forward(self, x):
        feat1 = self.pam1(x)
        feat2 = self.pam2(x)
        feat3 = self.pam3(x)
        feat4 = self.pam4(x)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, pool_size, norm_layer):
        super(PAM_Module, self).__init__()
        self.pool_size =pool_size
        self.chanel_in = in_dim
        self.key_dim = key_dim
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1), norm_layer(key_dim), nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1), norm_layer(key_dim), nn.ReLU())
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
        proj_query0 = self.query_conv(x)
        proj_query = proj_query0.view(m_batchsize, -1, width*height).permute(0, 2, 1) # n c h w   n hw c
        proj_key = self.key_conv(self.pool(x)).view(m_batchsize, -1, self.pool_size*self.pool_size) # n c s s  n c s^2
        energy = torch.bmm(proj_query, proj_key) # n hw s^2
        attention = self.softmax(energy)
        proj_value = proj_key

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.key_dim, height, width)

        out = out + proj_query0
        return out