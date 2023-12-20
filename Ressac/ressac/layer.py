"""
# File Name: layer.py
# Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import math
import numpy as np


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):
        # dims: 是一个包含三个整数的列表，表示输入维度(x_dim)、隐藏层维度(h_dim)和潜在空间维度(z_dim)。
        # bn: 是一个布尔值，表示是否在隐藏层中使用批量归一化(Batch Normalization)。默认为False，即不使用批量归一化
        # dropout: 是一个浮点数，表示在隐藏层中使用的dropout比例。默认为0，即不使用dropout。
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        # self.hidden = build_mlp([x_dim, *h_dim], bn=bn, dropout=dropout)
        # 这里调用了一个名为build_mlp的函数，它用于构建一个多层感知器(MLP)。
        # [x_dim]+h_dim将输入维度与隐藏层维度拼接成一个列表，作为build_mlp函数的输入，
        # 从而构建了一个包含多个隐藏层的MLP，并将其赋值给self.hidden，作为这个编码器的隐藏层。
        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        # 用于从高斯分布中采样潜在变量z。([x_dim]+h_dim)[-1]表示将输入维度和隐藏层维度拼接成一个列表，
        # 并取最后一个元素，即MLP输出层的维度。然后，用MLP的输出维度和潜在空间维度z_dim作为输入，
        # 创建了一个GaussianSample实例，并将其赋值给self.sample，表示这个编码器中的采样操作。
        self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)
        # self.sample = GaussianSample([x_dim, *h_dim][-1], z_dim)

    def forward(self, x):
        x = self.hidden(x);
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):
        # dims: 是一个包含三个整数的列表，表示潜在空间维度(z_dim)、隐藏层维度(h_dim)和输出维度(x_dim)。
        # bn: 是一个布尔值，表示是否在隐藏层中使用批量归一化(Batch Normalization)。默认为False，即不使用批量归一化。
        # dropout: 是一个浮点数，表示在隐藏层中使用的dropout比例。默认为0，即不使用dropout。
        # output_activation: 是一个PyTorch的激活函数实例，表示输出层的激活函数。默认为nn.Sigmoid()，即使用Sigmoid函数作为输出层的激活函数。
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()
        # z_dim:[10], h_dim:[], x_dim:[11236]
        [z_dim, h_dim, x_dim] = dims
        # [z_dim, *h_dim]将潜在空间维度与隐藏层维度拼接成一个列表，
        # 作为build_mlp函数的输入，从而构建了一个包含多个隐藏层的MLP，并将其赋值给self.hidden，作为这个解码器的隐藏层。
        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        # self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        # 创建了一个线性层(nn.Linear)，用于将隐藏层的输出映射到输出层。
        # [z_dim, *h_dim][-1]表示将潜在空间维度与隐藏层维度拼接成一个列表，并取最后一个元素，即MLP输出层的维度。
        # 然后，用MLP输出层的维度和输出维度x_dim作为输入，创建了一个线性层，并将其赋值给self.reconstruction。
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
        # self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)
        # 将传入的输出激活函数实例赋值给self.output_activation，表示这个解码器的输出层使用指定的激活函数。
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden(x)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)

class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

    def next(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


###################
###################
class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var

