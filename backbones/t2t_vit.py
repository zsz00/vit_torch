# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import numpy as np
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from .token_transformer import Token_transformer
from .transformer_block import Block, get_sinusoid_encoding


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            raise ValueError

        elif tokens_type == 'performer':
            raise ValueError

        elif tokens_type == 'face':
            self.soft_split0 = nn.Unfold(kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            n_patch = 196  # 196
            self.attention1 = Token_transformer(num_patch=n_patch*16, dim=in_chans * 5 * 5, in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)  # 3136
            self.attention2 = Token_transformer(num_patch=n_patch*4, dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)  # 768
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':
            raise ValueError

    def forward(self, x):
        x = self.soft_split0(x).transpose(1, 2)
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.soft_split1(x).transpose(1, 2)
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.soft_split2(x).transpose(1, 2)
        x = self.project(x)

        return x


class T2T_ViT(nn.Module):
    def __init__(self, tokens_type='face', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm1d):
        super().__init__()
        n_patch = 196  # 196
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tokens_to_token = T2T_module(tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=n_patch, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                num_patches=n_patch, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            self.blocks.append(block)
        self.norm = nn.BatchNorm1d(n_patch)

        # Classifier head
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim * n_patch, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.Linear(embed_dim, num_classes),
                nn.BatchNorm1d(num_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.tokens_to_token(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.num_classes > 0:
            x = self.head(x)
        return x
