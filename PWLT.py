from copy import deepcopy
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import einsum
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init, kaiming_init)
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, build_dropout, PatchEmbed, PatchMerging
from mmengine.registry import MODELS

# detection use
# from mmengine.runner.checkpoint import _load_checkpoint
# from ...utils import get_root_logger
# from collections import OrderedDict


class NewAttention(BaseModule):

    def __init__(self, token, embed_dims, num_heads=8, attn_drop=0., proj_drop=0., rate=None,
                 qkv_bias=True, dropout_layer=None, init_cfg=None):
        super(NewAttention, self).__init__(init_cfg)
        if dropout_layer is None:
            dropout_layer = dict(type='DropPath', drop_prob=0.)
        self.token = int(token),
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.rate = rate
        self.attend = nn.Sequential(
            nn.Softmax(-1),
            nn.Dropout(attn_drop)
        )
        if self.rate == 8:
            #ratio = 2*2:
            self.to_qkv2 = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3 // 2, bias = qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.norm2 = nn.LayerNorm(embed_dims)
            win2 = token // 2 if token % 2 == 0 else (token // 2 + 1)
            self.m2 = token // 2 * 2 if token % 2 == 0 else (token // 2 + 1) * 2
            self.up2 = nn.AdaptiveAvgPool2d((self.m2, self.m2))
            self.down2 = nn.AdaptiveAvgPool2d((token, token))
            self.sr2 = nn.AvgPool2d(kernel_size=win2, stride=win2)
            self.norm_qk2 = nn.LayerNorm(embed_dims // 2)
            self.g2_qk = nn.Sequential(
                nn.Linear(embed_dims // 2, embed_dims, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.relative_position_bias_table2 = nn.Parameter(torch.zeros((2*win2-1) ** 2, self.num_heads // 2))
            h2 = torch.arange(win2)
            w2 = torch.arange(win2)
            coords2 = torch.stack(torch.meshgrid([h2, w2]))
            coords2 = torch.flatten(coords2, 1)
            coords2 = coords2[:, :, None] - coords2[:, None, :]
            coords2 = coords2.permute(1, 2, 0).contiguous()
            coords2[:, :, 0] += win2-1
            coords2[:, :, 1] += win2-1
            coords2[:, :, 0] *= 2*win2 - 1
            coords2 = coords2.sum(-1)
            self.register_buffer("relative_index2", coords2)

            #ratio = 3*3:
            self.to_qkv3 = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3 // 2, bias = qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.norm3 = nn.LayerNorm(embed_dims)
            win3 = token // 3 if token % 3 == 0 else (token // 3 + 1)
            self.m3 = token // 3 * 3 if token % 3 == 0 else (token // 3 + 1) * 3
            self.up3 = nn.AdaptiveAvgPool2d((self.m3, self.m3))
            self.down3 = nn.AdaptiveAvgPool2d((token, token))
            self.sr3 = nn.AvgPool2d(kernel_size=win3, stride=win3)
            self.norm_qk3 = nn.LayerNorm(embed_dims // 2)
            self.g3_qk = nn.Sequential(
                nn.Linear(embed_dims // 2, embed_dims, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.relative_position_bias_table3 = nn.Parameter(torch.zeros((2*win3-1) ** 2, self.num_heads // 2))
            h3 = torch.arange(win3)
            w3 = torch.arange(win3)
            coords3 = torch.stack(torch.meshgrid([h3, w3]))
            coords3 = torch.flatten(coords3, 1)
            coords3 = coords3[:, :, None] - coords3[:, None, :]
            coords3 = coords3.permute(1, 2, 0).contiguous()
            coords3[:, :, 0] += win3-1
            coords3[:, :, 1] += win3-1
            coords3[:, :, 0] *= 2*win3 - 1
            coords3 = coords3.sum(-1)
            self.register_buffer("relative_index3", coords3)

        if self.rate == 4 or self.rate == 2:
            #ratio = 2*2:
            self.to_qkv2 = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3 // 4, bias = qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.norm2 = nn.LayerNorm(embed_dims)
            win2 = token // 2 if token % 2 == 0 else (token // 2 + 1)
            self.m2 = token // 2 * 2 if token % 2 == 0 else (token // 2 + 1) * 2
            self.up2 = nn.AdaptiveAvgPool2d((self.m2, self.m2))
            self.down2 = nn.AdaptiveAvgPool2d((token, token))
            self.sr2 = nn.AvgPool2d(kernel_size=win2, stride=win2)
            self.norm_qk2 = nn.LayerNorm(embed_dims // 4)
            self.g2_qk = nn.Sequential(
                nn.Linear(embed_dims // 4, embed_dims // 2, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.relative_position_bias_table2 = nn.Parameter(torch.zeros((2*win2-1) ** 2, self.num_heads // 4))
            h2 = torch.arange(win2)
            w2 = torch.arange(win2)
            coords2 = torch.stack(torch.meshgrid([h2, w2]))
            coords2 = torch.flatten(coords2, 1)
            coords2 = coords2[:, :, None] - coords2[:, None, :]
            coords2 = coords2.permute(1, 2, 0).contiguous()
            coords2[:, :, 0] += win2-1
            coords2[:, :, 1] += win2-1
            coords2[:, :, 0] *= 2*win2 - 1
            coords2 = coords2.sum(-1)
            self.register_buffer("relative_index2", coords2)

            #ratio = 3*3:
            self.to_qkv3 = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3 // 4, bias = qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.norm3 = nn.LayerNorm(embed_dims)
            win3 = token // 3 if token % 3 == 0 else (token // 3 + 1)
            self.m3 = token // 3 * 3 if token % 3 == 0 else (token // 3 + 1) * 3
            self.up3 = nn.AdaptiveAvgPool2d((self.m3, self.m3))
            self.down3 = nn.AdaptiveAvgPool2d((token, token))
            self.sr3 = nn.AvgPool2d(kernel_size=win3, stride=win3)
            self.norm_qk3 = nn.LayerNorm(embed_dims // 4)
            self.g3_qk = nn.Sequential(
                nn.Linear(embed_dims // 4, embed_dims // 2, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.relative_position_bias_table3 = nn.Parameter(torch.zeros((2*win3-1) ** 2, self.num_heads // 4))
            h3 = torch.arange(win3)
            w3 = torch.arange(win3)
            coords3 = torch.stack(torch.meshgrid([h3, w3]))
            coords3 = torch.flatten(coords3, 1)
            coords3 = coords3[:, :, None] - coords3[:, None, :]
            coords3 = coords3.permute(1, 2, 0).contiguous()
            coords3[:, :, 0] += win3-1
            coords3[:, :, 1] += win3-1
            coords3[:, :, 0] *= 2*win3 - 1
            coords3 = coords3.sum(-1)
            self.register_buffer("relative_index3", coords3)

            #ratio = 5*5:
            self.to_qkv5 = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3 // 2, bias = qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.norm5 = nn.LayerNorm(embed_dims)
            win5 = token // 5 if token % 5 == 0 else (token // 5 + 1)
            self.m5 = token // 5 * 5 if token % 5 == 0 else (token // 5 + 1) * 5
            self.up5 = nn.AdaptiveAvgPool2d((self.m5, self.m5))
            self.down5 = nn.AdaptiveAvgPool2d((token, token))
            self.sr5 = nn.AvgPool2d(kernel_size=win5, stride=win5)
            self.norm_qk5 = nn.LayerNorm(embed_dims // 2)
            self.g5_qk = nn.Sequential(
                nn.Linear(embed_dims // 2, embed_dims, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            self.relative_position_bias_table5 = nn.Parameter(torch.zeros((2*win5-1) ** 2, self.num_heads // 2))
            h5 = torch.arange(win5)
            w5 = torch.arange(win5)
            coords5 = torch.stack(torch.meshgrid([h5, w5]))
            coords5 = torch.flatten(coords5, 1)
            coords5 = coords5[:, :, None] - coords5[:, None, :]
            coords5 = coords5.permute(1, 2, 0).contiguous()
            coords5[:, :, 0] += win5-1
            coords5[:, :, 1] += win5-1
            coords5[:, :, 0] *= 2*win5 - 1
            coords5 = coords5.sum(-1)
            self.register_buffer("relative_index5", coords5)

        if self.rate == 1:
            self.to_qkv = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )

        self.proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims, bias=qkv_bias),
            nn.Dropout(proj_drop)
        )
        self.drop = build_dropout(dropout_layer)

    def forward(self, x, hw_shape):
        h, w = hw_shape
        B, N, C = x.shape
        if self.rate == 8:
            # print(self.token)
            #ratio=3
            H = self.m2
            W = self.m2
            win2 = H // 2
            qkv2 = self.to_qkv2(
                self.norm2(self.up2(x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()).reshape(B, -1, C))).reshape(
                B, H, W, C * 3 // 2)
            qkv2 = qkv2.reshape(B, H // win2, win2, W // win2, win2, 3, self.num_heads // 2, self.head_dim). \
                permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win2**2, self.num_heads // 2, win2**2,
                                                                     self.head_dim)
            bias2 = self.relative_position_bias_table2[self.relative_index2.view(-1)].view(win2**2, win2**2, -1)
            bias2 = bias2.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
            atte_l2 = self.attend((qkv2[0] @ qkv2[1].transpose(-2, -1).contiguous()) * self.scale + bias2)
            xl2 = (atte_l2 @ qkv2[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win2**2, win2**2, C // 2). \
                reshape(B, H // win2, W // win2, win2, win2, C // 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C // 2)
            vg2 = xl2.reshape(B, H // win2, win2, W // win2, win2, self.num_heads // 2, self.head_dim).permute(0, 5, 1, 3, 2, 4,
                                                                                                   6).contiguous(). \
                reshape(B, self.num_heads // 2, H * W // win2**2, win2**2, self.head_dim)
            qkg2 = self.g2_qk(self.norm_qk2(
                self.sr2(xl2.permute(0, 3, 1, 2).contiguous()).reshape(B, C // 2, -1).permute(0, 2, 1).contiguous())). \
                reshape(B, H * W // win2**2, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            atte_g2 = self.attend(qkg2[0] @ qkg2[1].transpose(-2, -1).contiguous() * self.scale)
            xg2 = einsum('b h i j, b h j m c -> b h i m c', atte_g2, vg2)
            xg2 = xg2.reshape(B, self.num_heads // 2, H // win2, W // win2, win2, win2, self.head_dim).permute(0, 2, 4, 3, 5, 1,
                                                                                                   6).contiguous(). \
                reshape(B, H, W, C // 2)
            xg2 = self.down2(xg2.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C // 2)
            H = self.m3
            W = self.m3
            win3 = H // 3
            qkv3 = self.to_qkv3(
                self.norm3(self.up3(x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()).reshape(B, -1, C))).reshape(
                B, H, W, C * 3 // 2)
            qkv3 = qkv3.reshape(B, H // win3, win3, W // win3, win3, 3, self.num_heads // 2, self.head_dim). \
                permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win3**2, self.num_heads // 2, win3**2,
                                                                     self.head_dim)
            bias3 = self.relative_position_bias_table3[self.relative_index3.view(-1)].view(win3**2, win3**2, -1)
            bias3 = bias3.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
            atte_l3 = self.attend((qkv3[0] @ qkv3[1].transpose(-2, -1).contiguous()) * self.scale + bias3)
            xl3 = (atte_l3 @ qkv3[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win3**2, win3**2, C // 2). \
                reshape(B, H // win3, W // win3, win3, win3, C // 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C // 2)
            vg3 = xl3.reshape(B, H // win3, win3, W // win3, win3, self.num_heads // 2, self.head_dim).permute(0, 5, 1, 3, 2, 4,
                                                                                                   6).contiguous(). \
                reshape(B, self.num_heads // 2, H * W // win3**2, win3**2, self.head_dim)
            qkg3 = self.g3_qk(self.norm_qk3(
                self.sr3(xl3.permute(0, 3, 1, 2).contiguous()).reshape(B, C // 2, -1).permute(0, 2, 1).contiguous())). \
                reshape(B, H * W // win3**2, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            atte_g3 = self.attend(qkg3[0] @ qkg3[1].transpose(-2, -1).contiguous() * self.scale)
            xg3 = einsum('b h i j, b h j m c -> b h i m c', atte_g3, vg3)
            xg3 = xg3.reshape(B, self.num_heads // 2, H // win3, W // win3, win3, win3, self.head_dim).permute(0, 2, 4, 3, 5, 1,
                                                                                                   6).contiguous(). \
                reshape(B, H, W, C // 2)
            xg3 = self.down3(xg3.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C // 2)
            x = torch.cat([xg2, xg3], dim=-1).contiguous()
        if self.rate == 4 or self.rate == 2:
            # print(self.token)
            H = self.m2
            W = self.m2
            win2 = H // 2
            qkv2 = self.to_qkv2(
                self.norm2(self.up2(x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()).reshape(B, -1, C))).reshape(
                B, H, W, C * 3 // 4)
            qkv2 = qkv2.reshape(B, H // win2, win2, W // win2, win2, 3, self.num_heads // 4, self.head_dim). \
                permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win2**2, self.num_heads // 4, win2**2,
                                                                     self.head_dim)
            bias2 = self.relative_position_bias_table2[self.relative_index2.view(-1)].view(win2**2, win2**2, -1)
            bias2 = bias2.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
            atte_l2 = self.attend((qkv2[0] @ qkv2[1].transpose(-2, -1).contiguous()) * self.scale + bias2)
            xl2 = (atte_l2 @ qkv2[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win2**2, win2**2, C // 4). \
                reshape(B, H // win2, W // win2, win2, win2, C // 4).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C // 4)
            vg2 = xl2.reshape(B, H // win2, win2, W // win2, win2, self.num_heads // 4, self.head_dim).permute(0, 5, 1, 3, 2, 4,
                                                                                                   6).contiguous(). \
                reshape(B, self.num_heads // 4, H * W // win2**2, win2**2, self.head_dim)
            qkg2 = self.g2_qk(self.norm_qk2(
                self.sr2(xl2.permute(0, 3, 1, 2).contiguous()).reshape(B, C // 4, -1).permute(0, 2, 1).contiguous())). \
                reshape(B, H * W // win2**2, 2, self.num_heads // 4, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            atte_g2 = self.attend(qkg2[0] @ qkg2[1].transpose(-2, -1).contiguous() * self.scale)
            xg2 = einsum('b h i j, b h j m c -> b h i m c', atte_g2, vg2)
            xg2 = xg2.reshape(B, self.num_heads // 4, H // win2, W // win2, win2, win2, self.head_dim).permute(0, 2, 4, 3, 5, 1,
                                                                                                   6).contiguous(). \
                reshape(B, H, W, C // 4)
            xg2 = self.down2(xg2.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C // 4)

            #ratio=3
            H = self.m3
            W = self.m3
            win3 = H // 3
            qkv3 = self.to_qkv3(
                self.norm3(self.up3(x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()).reshape(B, -1, C))).reshape(
                B, H, W, C * 3 // 4)
            qkv3 = qkv3.reshape(B, H // win3, win3, W // win3, win3, 3, self.num_heads // 4, self.head_dim). \
                permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win3**2, self.num_heads // 4, win3**2,
                                                                     self.head_dim)
            bias3 = self.relative_position_bias_table3[self.relative_index3.view(-1)].view(win3**2, win3**2, -1)
            bias3 = bias3.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
            atte_l3 = self.attend((qkv3[0] @ qkv3[1].transpose(-2, -1).contiguous()) * self.scale + bias3)
            xl3 = (atte_l3 @ qkv3[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win3**2, win3**2, C // 4). \
                reshape(B, H // win3, W // win3, win3, win3, C // 4).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C // 4)
            vg3 = xl3.reshape(B, H // win3, win3, W // win3, win3, self.num_heads // 4, self.head_dim).permute(0, 5, 1, 3, 2, 4,
                                                                                                   6).contiguous(). \
                reshape(B, self.num_heads // 4, H * W // win3**2, win3**2, self.head_dim)
            qkg3 = self.g3_qk(self.norm_qk3(
                self.sr3(xl3.permute(0, 3, 1, 2).contiguous()).reshape(B, C // 4, -1).permute(0, 2, 1).contiguous())). \
                reshape(B, H * W // win3**2, 2, self.num_heads // 4, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            atte_g3 = self.attend(qkg3[0] @ qkg3[1].transpose(-2, -1).contiguous() * self.scale)
            xg3 = einsum('b h i j, b h j m c -> b h i m c', atte_g3, vg3)
            xg3 = xg3.reshape(B, self.num_heads // 4, H // win3, W // win3, win3, win3, self.head_dim).permute(0, 2, 4, 3, 5, 1,
                                                                                                   6).contiguous(). \
                reshape(B, H, W, C // 4)
            xg3 = self.down3(xg3.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C // 4)

            #ratio=5
            H = self.m5
            W = self.m5
            win5 = H // 5
            qkv5 = self.to_qkv5(
                self.norm5(self.up5(x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()).reshape(B, -1, C))).reshape(
                B, H, W, C * 3 // 2)
            qkv5 = qkv5.reshape(B, H // win5, win5, W // win5, win5, 3, self.num_heads // 2, self.head_dim). \
                permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win5**2, self.num_heads // 2, win5**2,
                                                                     self.head_dim)
            bias5 = self.relative_position_bias_table5[self.relative_index5.view(-1)].view(win5**2, win5**2, -1)
            bias5 = bias5.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
            atte_l5 = self.attend((qkv5[0] @ qkv5[1].transpose(-2, -1).contiguous()) * self.scale + bias5)
            xl5 = (atte_l5 @ qkv5[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win5**2, win5**2, C // 2). \
                reshape(B, H // win5, W // win5, win5, win5, C // 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C // 2)
            vg5 = xl5.reshape(B, H // win5, win5, W // win5, win5, self.num_heads // 2, self.head_dim).permute(0, 5, 1, 3, 2, 4,
                                                                                                   6).contiguous(). \
                reshape(B, self.num_heads // 2, H * W // win5**2, win5**2, self.head_dim)
            qkg5 = self.g5_qk(self.norm_qk5(
                self.sr5(xl5.permute(0, 3, 1, 2).contiguous()).reshape(B, C // 2, -1).permute(0, 2, 1).contiguous())). \
                reshape(B, H * W // win5**2, 2, self.num_heads // 2, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            atte_g5= self.attend(qkg5[0] @ qkg5[1].transpose(-2, -1).contiguous() * self.scale)
            xg5 = einsum('b h i j, b h j m c -> b h i m c', atte_g5, vg5)
            xg5 = xg5.reshape(B, self.num_heads // 2, H // win5, W // win5, win5, win5, self.head_dim).permute(0, 2, 4, 3, 5, 1,
                                                                                                   6).contiguous(). \
                reshape(B, H, W, C // 2)
            xg5 = self.down5(xg5.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C // 2)

            x = torch.cat([xg2, xg3, xg5], dim=-1).contiguous()

        if self.rate == 1:
            # print(self.token)
            qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            attn = self.attend((qkv[0] @ qkv[1].transpose(-2, -1).contiguous()) * self.scale)
            x = (attn @ qkv[2]).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)

        return self.drop(self.proj(x))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, hw_shape):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, hw_shape))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NewBlock(BaseModule):

    def __init__(self,
                 token,
                 embed_dims,
                 num_heads,
                 rate,
                 feedforward_ratio,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(NewBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = NewAttention(
            token=token,
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            rate=rate,
            qkv_bias=qkv_bias,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        # self.ffn = FFN(
        #     embed_dims=embed_dims,
        #     feedforward_channels=int(embed_dims * feedforward_ratio),
        #     num_fcs=2,
        #     ffn_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     act_cfg=act_cfg,
        #     add_identity=True,
        #     init_cfg=None)
        self.ffn = Mlp(embed_dims, hidden_features=int(embed_dims * feedforward_ratio), drop=drop_rate)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape) + identity
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, hw_shape) + identity
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class NewBlockSequence(BaseModule):

    def __init__(self,
                 token,
                 embed_dims,
                 num_heads,
                 rate,
                 feedforward_ratio,
                 depth,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        self.pos_block = PosCNN(embed_dims, embed_dims)
        for i in range(depth):
            block = NewBlock(
                token=token,
                embed_dims=embed_dims,
                num_heads=num_heads,
                rate=rate,
                feedforward_ratio=feedforward_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for i, block in enumerate(self.blocks):
            x = block(x, hw_shape)
            if i == 0:
                x = self.pos_block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class DW_Pwt3(BaseModule):

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 rates=(8, 4, 2, 1),
                 patch_sizes=(7, 3, 3, 3),
                 mlp_ratio=(8, 8, 4, 4),
                 depths=(1, 1, 4, 2),
                 num_heads=(2, 4, 8, 16),
                 strides=(4, 2, 2, 2),
                 paddings=(3, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(DW_Pwt3, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        # assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        row = pretrain_img_size[0] // strides[0]
        # col = pretrain_img_size[1] // strides[0]
        # self.tokens = row * col
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_sizes[0]
            patch_col = pretrain_img_size[1] // patch_sizes[0]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, embed_dims, patch_row, patch_col)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=patch_sizes[i+1],
                    stride=strides[i + 1],
                    padding=paddings[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = NewBlockSequence(
                token= int(row // int(math.pow(2, i)) if row % int(math.pow(2, i)) == 0 else row // int(math.pow(2, i)) + 1),
                embed_dims=in_channels,
                num_heads=num_heads[i],
                rate=rates[i],
                feedforward_ratio=mlp_ratio[i],
                depth=depths[i],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DW_Pwt3, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        # logger = get_root_logger()
        if self.init_cfg is None:
            # logger.warn(f'No pre-trained weights for '
            #             f'{self.__class__.__name__}, '
            #             f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        # else:
        #     assert 'checkpoint' in self.init_cfg, f'Only support ' \
        #                                           f'specify `Pretrained` in ' \
        #                                           f'`init_cfg` in ' \
        #                                           f'{self.__class__.__name__} '
        #     ckpt = _load_checkpoint(
        #         self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        #     if 'state_dict' in ckpt:
        #         _state_dict = ckpt['state_dict']
        #     elif 'model' in ckpt:
        #         _state_dict = ckpt['model']
        #     else:
        #         _state_dict = ckpt
        #     # if self.convert_weights:
        #     #     # supported loading weight from original repo,
        #     #     _state_dict = swin_converter(_state_dict)
        #
        #     state_dict = OrderedDict()
        #     for k, v in _state_dict.items():
        #         if k.startswith('backbone.'):
        #             state_dict[k[9:]] = v
        #
        #     # strip prefix of state_dict
        #     if list(state_dict.keys())[0].startswith('module.'):
        #         state_dict = {k[7:]: v for k, v in state_dict.items()}
        #
        #     # reshape absolute position embedding
        #     if state_dict.get('absolute_pos_embed') is not None:
        #         absolute_pos_embed = state_dict['absolute_pos_embed']
        #         N1, L, C1 = absolute_pos_embed.size()
        #         N2, C2, H, W = self.absolute_pos_embed.size()
        #         if N1 != N2 or C1 != C2 or L != H * W:
        #             logger.warning('Error in loading absolute_pos_embed, pass')
        #         else:
        #             state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
        #                 N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
        #
        #     # interpolate position bias table if needed
        #     relative_position_bias_table_keys = [
        #         k for k in state_dict.keys()
        #         if 'relative_position_bias_table' in k
        #     ]
        #     for table_key in relative_position_bias_table_keys:
        #         table_pretrained = state_dict[table_key]
        #         table_current = self.state_dict()[table_key]
        #         L1, nH1 = table_pretrained.size()
        #         L2, nH2 = table_current.size()
        #         if nH1 != nH2:
        #             logger.warning(f'Error in loading {table_key}, pass')
        #         elif L1 != L2:
        #             S1 = int(L1 ** 0.5)
        #             S2 = int(L2 ** 0.5)
        #             table_pretrained_resized = F.interpolate(
        #                 table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
        #                 size=(S2, S2),
        #                 mode='bicubic')
        #             state_dict[table_key] = table_pretrained_resized.view(
        #                 nH2, L2).permute(1, 0).contiguous()
        #
        #     # load state_dict
        #     self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            h, w = self.absolute_pos_embed.shape[1:3]
            if hw_shape[0] != h or hw_shape[1] != w:
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed,
                    size=hw_shape,
                    mode='bicubic',
                    align_corners=False).flatten(2).transpose(1, 2)
            else:
                absolute_pos_embed = self.absolute_pos_embed.flatten(
                    2).transpose(1, 2)
            x = x + absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return tuple(outs)

