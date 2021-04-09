
from .iresnet import iresnet34, iresnet50, iresnet100
from .t2t_vit import T2T_ViT
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

#from timm.models.helpers import load_pretrained
#from timm.models.registry import register_model
#from timm.models.layers import trunc_normal_
#import numpy as np
#from .token_transformer import Token_transformer
#from .token_performer import Token_performer
#from .transformer_block import Block, get_sinusoid_encoding

class Face_DeiT(nn.Module):
    def __init__(self, img_size=112):
        super().__init__()
        ch = 16
        self.cnn = nn.Sequential(
                iresnet50(False, dropout=0, output_featmap=True),
                nn.Conv2d(64, ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ch, eps=1e-05),
                nn.PReLU(ch),
                )
        embed_dim = 768
        num_classes = 512
        self.trans = nn.Sequential(
                VisionTransformer(
                    img_size=img_size//2, patch_size=8, embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=ch, num_classes=0)
                )
        #model.default_cfg = _cfg()
        self.head = nn.Sequential(
                nn.Linear(embed_dim, num_classes),
                nn.BatchNorm1d(num_classes),
                #nn.BatchNorm1d(512, affine=False),
                #nn.BatchNorm1d(512),
                )


        for m in self.modules():
            #if isinstance(m, nn.Conv2d):
            #    #nn.init.normal_(m.weight, 0, 0.1)
            #    nn.init.kaiming_normal_( m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                #nn.init.normal(m.weight, std=.01)
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.cnn(x)
        x = self.trans(x)
        x = self.head(x)
        #print(x.shape)
        #print(len(x))
        #x = self.bn(x)
        #x = self.backbone_cnn(x)
        #x = self.backbone_vit(x)
        return x

