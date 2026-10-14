import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat

import os
from matplotlib import pyplot as plt

class ChannelTransfer(nn.Module):
    r"""Transfer 2D feature channel"""

    def __init__(self, n_feats=16):
        super().__init__()
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x

class ChannelReturn(nn.Module):
    r"""Return 2D feature channel"""

    def __init__(self, n_feats=16):
        super().__init__()
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class SFS2D(nn.Module):
    r"""2D Selective Fusion Scan"""
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act1 = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))  # (K=4, N, inner)
        del self.x_proj1

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.dt_projs1 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0))  # (K=4, inner)
        del self.dt_projs1

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.fusion_proj = nn.Linear(self.d_inner * 2, self.d_inner, bias=bias, **factory_kwargs)
        self.fusion_proj1 = nn.Linear(self.d_inner * 2, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm1 = nn.LayerNorm(self.d_inner)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, mode=0):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, 4, c, h*w)
        if mode == 0:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight1)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        if mode == 0:
            Ds = self.Ds.float().view(-1)
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
        else:
            Ds = self.Ds1.float().view(-1)
            As = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)
            dt_projs_bias = self.dt_projs_bias1.float().view(-1)  # (k * d)
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, x1: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # 2 linear
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) #b c h*w

        x1 = self.in_proj1(x1)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.act1(self.conv2d1(x1)) #b c h*w

        # 2D-SSM
        y1, y2, y3, y4 = self.forward_core(x, 0)
        y11, y21, y31, y41 = self.forward_core(x1, 1)

        assert y1.dtype == torch.float32
        v = (y1 + y2).view(B, -1, H, W)
        h = (y3 + y4).view(B, -1, H, W)
        y = self.fusion_proj(torch.cat((v, h), 1).permute(0, 2, 3, 1).contiguous())

        v1 = (y1 + y2 + y11 + y21).view(B, -1, H, W)
        h1 = (y3 + y4 + y31 + y41).view(B, -1, H, W)
        y1 = self.fusion_proj1(torch.cat((v1, h1), 1).permute(0, 2, 3, 1).contiguous())
        # self.show_feature_map(H, y1.permute(0, 3, 1, 2), 'Attention/net')

        y = self.out_norm(y)
        y1 = self.out_norm1(y1)

        y = y * F.silu(z)
        y1 = y1 * F.silu(z)
        out = self.out_proj(y + y1)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def show_feature_map(self, size, feature_map, path):
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
                                       feature_map.shape[2])
        upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))
        feature_map = upsample(feature_map)
        feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        feature_map_num = feature_map.shape[0]
        for index in range(1, feature_map_num + 1):
            plt.axis('off')
            folder_name = path
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.imsave(folder_name + '//' + str(size) + "_" + str(index) + ".png",
                       feature_map[index - 1].detach().cpu().numpy(), cmap='jet')

class DAFM(nn.Module):
    r"""Depth-aware Fusion Mamba"""
    def __init__(
            self,
            dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super(DAFM, self).__init__()
        self.cf1 = ChannelTransfer(dim)
        self.cf2 = ChannelTransfer(dim)
        self.sfs2d = SFS2D(d_model=dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop = DropPath(drop_path)
        self.cr = ChannelReturn(dim)

    def forward(self, x, d):
        d = F.interpolate(d, size=[x.size(2), x.size(3)], mode='nearest')
        xd = x * d
        x1 = self.cf1(x)
        xd = self.cf2(xd)
        x1 = self.sfs2d(x1, xd)
        x1 = self.drop(x1)
        out = x + self.cr(x1)

        return out