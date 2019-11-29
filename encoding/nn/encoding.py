##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Package Core NN Modules."""
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Module, Parameter
import torch.nn as nn
from ..functions import scaled_l2, aggregate

__all__ = ['Encoding', 'dict_Encoding', 'pydict_Encoding', 'simple_pydict_Encoding']

class Encoding(Module):
    r"""
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\times H\times W`
    as a set of C-dimentional input features :math:`X=\{x_1, ...x_N\}`, where N is total number
    of features given by :math:`H\times W`, which learns an inherent codebook
    :math:`D=\{d_1,...d_K\}` and a set of smoothing factor of visual centers
    :math:`S=\{s_1,...s_K\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\{e_1,...e_K\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'



class dict_Encoding(Module):
    
    def __init__(self, D, K):
        super(dict_Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(D, K), requires_grad=True)
        # self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.gamma = Parameter(torch.zeros(1))
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        # self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        h,w = X.size(2), X.size(3)
        if X.dim() == 3:
            # BxDxN => BxNxD
            Xt = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            Xt = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(torch.matmul(Xt,self.codewords),dim=-1)
        # aggregate
        E = X + self.gamma*torch.matmul(A, self.codewords.transpose(0,1)).permute(0,2,1).view(B,D,h,w)
        return E


class pydict_Encoding(Module):

    def __init__(self, in_dim, norm_layer, D, K1,K2,K3):
        super(pydict_Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K1, self.K2, self.K3 = D, K1, K2, K3
        self.codewords1 = Parameter(torch.Tensor(D, K1), requires_grad=True)
        self.codewords2 = Parameter(torch.Tensor(D, K2), requires_grad=True)
        self.codewords3 = Parameter(torch.Tensor(D, K3), requires_grad=True)

        self.scale1 = Parameter(torch.Tensor(K1), requires_grad=True)
        self.scale2 = Parameter(torch.Tensor(K2), requires_grad=True)
        self.scale3 = Parameter(torch.Tensor(K3), requires_grad=True)


        self.gamma1 = Parameter(torch.zeros(1),requires_grad=True)
        self.gamma2 = Parameter(torch.zeros(1),requires_grad=True)
        self.gamma3 = Parameter(torch.zeros(1),requires_grad=True)
        self.conv1= nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, padding=0, bias=False),
                                    norm_layer(in_dim), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, padding=0, bias=False),
                                   norm_layer(in_dim), nn.ReLU(True))
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K1 * self.D) ** (1 / 2))
        self.codewords1.data.uniform_(-std1, std1)

        std2 = 1. / ((self.K2 * self.D) ** (1 / 2))
        self.codewords2.data.uniform_(-std2, std2)

        std3 = 1. / ((self.K3 * self.D) ** (1 / 2))
        self.codewords3.data.uniform_(-std3, std3)

        self.scale1.data.uniform_(-1, 0)
        self.scale2.data.uniform_(-1, 0)
        self.scale3.data.uniform_(-1, 0)


    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        h, w = X.size(2), X.size(3)
        X2 = self.conv1(X)
        X3 = self.conv2(X2)

        Xt1 = X.view(B, D, -1)
        Xt2 = X2.view(B, D, -1)
        Xt3 = X3.view(B, D, -1)

        # assignment weights BxNxK
        A1 = torch.matmul((self.scale1*self.codewords1).permute(1, 0), Xt1)
        A2 = torch.matmul((self.scale2*self.codewords2).permute(1, 0), Xt2)
        A3 = torch.matmul((self.scale3*self.codewords3).permute(1, 0), Xt3)

        S2 = F.softmax(torch.matmul(self.codewords1.permute(1, 0), self.codewords2), dim=-1)
        S3 = F.softmax(torch.matmul(self.codewords2.permute(1, 0), self.codewords3), dim=-1)

        # aggregate
        SA3 = torch.matmul(S3, A3)
        SA2 = torch.matmul(S2, A2+self.gamma3*SA3)
        SA1 = torch.matmul(self.codewords1, F.softmax((A1 + self.gamma2*SA2), dim=-2))

        E = X + self.gamma1*SA1.view(B,D,h,w)

        return E


class simple_pydict_Encoding(Module):

    def __init__(self, in_dim,norm_layer, D, K1, K2, K3):
        super(simple_pydict_Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K1, self.K2, self.K3 = D, K1, K2, K3
        self.codewords1 = Parameter(torch.Tensor(D, K1), requires_grad=True)
        self.codewords2 = Parameter(torch.Tensor(D, K2), requires_grad=True)
        self.codewords3 = Parameter(torch.Tensor(D, K3), requires_grad=True)

        self.scale1 = Parameter(torch.Tensor(K1), requires_grad=True)
        self.scale2 = Parameter(torch.Tensor(K2), requires_grad=True)
        self.scale3 = Parameter(torch.Tensor(K3), requires_grad=True)

        self.gamma1 = Parameter(torch.zeros(1), requires_grad=True)
        self.gamma2 = Parameter(torch.zeros(1), requires_grad=True)
        self.gamma3 = Parameter(torch.zeros(1), requires_grad=True)

        self.conv1= nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, padding=0, bias=False),
                                    norm_layer(in_dim), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, padding=0, bias=False),
                                   norm_layer(in_dim), nn.ReLU(True))
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K1 * self.D) ** (1 / 2))
        self.codewords1.data.uniform_(-std1, std1)

        std2 = 1. / ((self.K2 * self.D) ** (1 / 2))
        self.codewords2.data.uniform_(-std2, std2)

        std3 = 1. / ((self.K3 * self.D) ** (1 / 2))
        self.codewords3.data.uniform_(-std3, std3)

        self.scale1.data.uniform_(-1, 0)
        self.scale2.data.uniform_(-1, 0)
        self.scale3.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        h, w = X.size(2), X.size(3)

        X2 = self.conv1(X)
        X3 = self.conv2(X2)

        Xt1 = X.view(B, D, -1).transpose(1, 2).contiguous()
        Xt2 = X2.view(B, D, -1).transpose(1, 2).contiguous()
        Xt3 = X3.view(B, D, -1).transpose(1, 2).contiguous()

        # assignment weights BxNxK
        A1 = F.softmax(torch.matmul(Xt1, self.codewords1), dim=-1)
        A2 = F.softmax(torch.matmul(Xt2, self.codewords2), dim=-1)
        A3 = F.softmax(torch.matmul(Xt3, self.codewords3), dim=-1)

        # aggregate
        E1 = self.gamma1 * torch.matmul(A1, self.codewords1.transpose(0,1)).permute(0,2,1).view(B,D,h,w)
        E2 = self.gamma2 * torch.matmul(A2, self.codewords2.transpose(0, 1)).permute(0, 2, 1).view(B, D, h, w)
        E3 = self.gamma3 * torch.matmul(A3, self.codewords3.transpose(0, 1)).permute(0, 2, 1).view(B, D, h, w)
        E = X + E1 + E2 + E3
        return E
