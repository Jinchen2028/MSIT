import torch
from torch import nn as nn
from torch.nn import functional as f

import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class RiM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RiM, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        self.linear = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=1, stride=1,
                                 groups=G, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + (self.convmap(origin_weight) * self.linear(origin_weight)).view(*self.origin_kernel_shape)
        return f.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)

class LEM(nn.Module):
    r"""Mixing pixels with their surrounding pixels.

    Args:
        planes (int):
        mix_margin (int):
        mix_mode (str):

    Warnings:
        The padding operation may result in incorrect edge textures
        when using a larger mix_margin.

    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(LEM, self).__init__()

        assert planes % 8 == 0

        self.planes = planes
        self.mix_margin = nn.Parameter(torch.tensor(mix_margin, dtype=torch.float), requires_grad=True)

        self.mask = nn.Parameter(torch.zeros((self.planes, 1, 3, 3)), requires_grad=False)

        self.mask[3::4, 0, 0, mix_margin] = 1.
        self.mask[2::4, 0, -1, mix_margin] = 1.
        self.mask[1::4, 0, mix_margin, 0] = 1.
        self.mask[0::4, 0, mix_margin, -1] = 1.

        self.mask[4::8, 0, 0, 2] = 1.
        self.mask[5::8, 0, 2, 0] = 1.
        self.mask[6::8, 0, 2, 2] = 1.
        self.mask[7::8, 0, 0, 0] = 1.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        m = int(self.mix_margin.item())
        x = f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(m, m), groups=self.planes)

        return x

class MsC(nn.Module):
    def __init__(self, midc, num = 4):
        super().__init__()

        self.headc = midc // num
        self.num = num
        self.midc = midc
        self.proj1 = nn.Conv2d(midc , midc * 2, kernel_size=1, padding=0, stride=1,
                               groups=self.headc)
        print(f'MsSA_q: {self.num}')
        self.proj2 = nn.Conv2d(midc * 2, midc , 1)
        self.proj0 = nn.Conv2d(midc, midc, 1)
        self.bn = nn.BatchNorm2d(midc * 2)
        self.act = nn.GELU()

        for i in range(self.num):
            # local_conv = nn.Conv2d(self.headc, self.headc, kernel_size=(3 + i * 2),
            #                        padding=(1 + i), stride=1)
            local_conv = RiM(self.headc, self.headc, kernel_size=(3 + i * 2),\
                                stride=1, padding=(1 + i), groups=1, map_k=3)
            setattr(self, f"local_conv_{i + 1}", local_conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feat, name='0'):

        x = torch.split(self.proj0(feat), self.headc, dim = 1)
        out = []
        for i in range(self.num):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            x_i = x[i]
            out_ = local_conv(x_i)
            out.append(out_)

        s_out = torch.cat([*out],dim = 1)
        out = self.proj2(self.act(self.bn(self.proj1(s_out))))

        return out

class amsno(nn.Module):
    def __init__(self, dim, ca_num_heads=16, qkv_bias=False, proj_drop=0.,
                 ca_attention=1, expand_ratio=2, use_lem = True):
        super().__init__()

        self.ca_attention = ca_attention
        self.lem = LEM(planes=dim)
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.use_lem = use_lem


        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        print(f'MsC_heads: {self.ca_num_heads}')
        print(f'use_lem: {self.use_lem}')

        self.act = nn.GELU()

        self.proj = nn.Conv2d(dim, dim , 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            self.v = nn.Conv2d(dim, dim , 1, bias=qkv_bias)
            self.s = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                # Ms_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                #                        padding=(1 + i), stride=1)
                Ms_conv = RiM(dim // self.ca_num_heads, dim // self.ca_num_heads, \
                                    kernel_size=(3 + i * 2), stride=1, padding=(1 + i), groups=1, map_k=3)
                setattr(self, f"Ms_conv_{i + 1}", Ms_conv)
            self.sim0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.ca_num_heads)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)
            self.sim1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            if self.use_lem:
                s = torch.split(self.lem(self.s(x)), C // self.ca_num_heads, dim=1)
            else:
                s = torch.split(self.s(x), C // self.ca_num_heads, dim=1)

            for i in range(self.ca_num_heads):
                Ms_conv = getattr(self, f"Ms_conv_{i + 1}")
                s_i = s[i]
                s_i = Ms_conv(s_i)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out, s_i], 1)

            s_out = self.sim1(self.act(self.bn(self.sim0(s_out))))
            x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)

        return x