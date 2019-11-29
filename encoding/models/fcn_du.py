###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import upsample
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['FCN_DU', 'get_fcn_du', 'get_fcn_resnet50_pcontext', 'get_fcn_resnet50_ade']


class FCN_DU(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """

    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN_DU, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCN_DUHead(512, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, bias=False),
                                   norm_layer(128),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(2048, 2048, 3, padding=1, bias=False),
                                   norm_layer(2048),
                                   nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1, bias=False),
                                   norm_layer(1024),
                                   nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                   norm_layer(128),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        # rearr_c1 = self.conv2(c1)
        rearr_c2 = self.conv3(c2)
        rearr_c3 = self.conv4(c3)
        rearr_c4 = self.conv5(c4)
        c4_du = self.conv6(F.pixel_shuffle(rearr_c4, 2) + rearr_c3)
        c3_du = self.conv7(F.pixel_shuffle(c4_du, 2) + rearr_c2)
        # c2_du = self.conv8(F.pixel_shuffle(c3_du, 2) + rearr_c1)

        x = self.head(c3_du)
        x = upsample(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class FCN_DUHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCN_DUHead, self).__init__()
        self.conv5 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


def get_fcn_du(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = FCN_DU(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


def get_fcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)


def get_fcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn('ade20k', 'resnet50', pretrained, root=root, **kwargs)
