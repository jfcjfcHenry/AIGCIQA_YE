import math
import torch
from torch import einsum, nn
import numpy as np
from functools import partial
import torch.nn.functional as F
from torch.nn import Softmin

class FactorAtt_ConvRelPosEnc(nn.Module):
    """带有卷积相对位置编码的因子化注意力类"""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 共享的卷积相对位置编码
        self.crpe = shared_crpe

    def forward(self, q, k, v, minus=True):
        B, N, C = q.shape

        # 生成Q, K, V
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 因子化注意力
        use_efficient = minus
        if use_efficient:
            k_softmax = k.softmax(dim=2)
            k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)
        else:
            k_softmax = k.softmax(dim=2)
            k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)

        x = factor_att if use_efficient else v - factor_att
        x = x.transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Mlp(nn.Module):
    """前馈神经网络（FFN, 或称 MLP）类"""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MHCABlock(nn.Module):
    """多头卷积自注意力块"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.fuse = nn.Linear(dim * 2, dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.norm2 = norm_layer(dim)

    def forward(self, q, k, v, minus=True):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        x = q + self.factoratt_crpe(q, k, v, minus)
        cur = self.norm2(x)
        x = x + self.mlp(cur)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
if __name__ == '__main__':
    dim = 64
    num_heads = 8  # 指定头数

    block = MHCABlock(dim=dim, num_heads=num_heads)

    # 输入尺寸
    B, C, H, W = 1, dim, 7, 7
    q = torch.rand(B, C, H, W)
    k = torch.rand(B, C, H, W)
    v = torch.rand(B, C, H, W)

    # 前向传播
    output = block(q, k, v)
    print(f"输入尺寸: {q.size()}")
    print(f"输出尺寸: {output.size()}")
