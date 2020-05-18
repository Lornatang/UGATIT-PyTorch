# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # Initial convolution block
        down_layers = [nn.ReflectionPad2d(3),
                       nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0, bias=False),
                       nn.InstanceNorm2d(64),
                       nn.ReLU(inplace=True)]

        # Downsampling
        down_layers += [nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 128, 3, stride=2, padding=0, bias=False),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 256, 3, stride=2, padding=0, bias=False),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(inplace=True)]

        # Down sampling residual blocks
        for _ in range(4):
            down_layers += [ResNetBlock(256)]

        # Class Activation Map
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Gamma, Beta block
        fc = nn.Sequential(nn.Linear(64 * 64 * 256, 256, bias=False),
                           nn.ReLU(True),
                           nn.Linear(256, 256, bias=False),
                           nn.ReLU(True))
        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        # Up sampling residual blocks
        for i in range(4):
            setattr(self, "ResNetAdaILNBlock_" + str(i + 1), ResNetAdaILNBlock(256))

        # Upsampling
        up_layers = [nn.Upsample(scale_factor=2, mode="nearest"),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
                     ILN(128),
                     nn.ReLU(True),

                     nn.Upsample(scale_factor=2, mode="nearest"),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
                     ILN(64),
                     nn.ReLU(True),

                     nn.ReflectionPad2d(3),
                     nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.down_layers = nn.Sequential(*down_layers)
        self.fc = nn.Sequential(*fc)
        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, inputs):
        x = self.down_layers(inputs)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x_ = self.fc(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(4):
            x = getattr(self, "ResNetAdaILNBlock_" + str(i + 1))(x, gamma, beta)
        out = self.up_layers(x)

        return out, cam_logit, heatmap


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        out = x + self.main(x)
        return out


class ResNetAdaILNBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class AdaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor([1, num_features, 1, 1]))
        self.rho.data.fill_(0.9)

    def forward(self, inputs, gamma, beta):
        in_mean, in_var = torch.mean(inputs, dim=[2, 3], keepdim=True), torch.var(inputs, dim=[2, 3], keepdim=True)
        out_in = (inputs - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(inputs, dim=[1, 2, 3], keepdim=True), torch.var(inputs, dim=[1, 2, 3],
                                                                                     keepdim=True)
        out_ln = (inputs - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(inputs.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(inputs.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor([1, num_features, 1, 1]))
        self.gamma = Parameter(torch.Tensor([1, num_features, 1, 1]))
        self.beta = Parameter(torch.Tensor([1, num_features, 1, 1]))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, inputs):
        in_mean, in_var = torch.mean(inputs, dim=[2, 3], keepdim=True), torch.var(inputs, dim=[2, 3], keepdim=True)
        out_in = (inputs - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(inputs, dim=[1, 2, 3], keepdim=True), torch.var(inputs, dim=[1, 2, 3],
                                                                                     keepdim=True)
        out_ln = (inputs - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(inputs.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(inputs.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(inputs.shape[0], -1, -1, -1) + self.beta.expand(inputs.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, n_layers):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            channel_ratio = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(64 * channel_ratio, 64 * channel_ratio * 2, kernel_size=4, stride=2, padding=0,
                                    bias=True)),
                      nn.LeakyReLU(0.2, True)]

        channel_ratio = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(64 * channel_ratio, 64 * channel_ratio * 2, kernel_size=4, stride=1, padding=0,
                                bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        channel_ratio = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(64 * channel_ratio, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(64 * channel_ratio, 1, bias=False))
        self.conv1x1 = nn.Conv2d(64 * channel_ratio * 2, 64 * channel_ratio, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(64 * channel_ratio, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        x = self.model(inputs)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max
        assert clip_min < clip_max

    def __call__(self, module):
        if hasattr(module, "rho"):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
