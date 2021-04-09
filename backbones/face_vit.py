from .iresnet import iresnet34, iresnet50, iresnet100
from .t2t_vit import T2T_ViT
import torch
import torch.nn as nn

# from timm.models.helpers import load_pretrained
# from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

# import numpy as np
# from .token_transformer import Token_transformer
# from .token_performer import Token_performer
# from .transformer_block import Block, get_sinusoid_encoding


class Face_ViT(nn.Module):
    def __init__(self, img_size=112):
        super().__init__()
        self.use_cnn = True
        embed_dim = 768
        num_classes = 512
        print('init face_vit with:', self.use_cnn, embed_dim, num_classes)
        if self.use_cnn:
            # self.cnn = iresnet50(False, dropout=0, output_featmap=True)
            ch = 16
            self.cnn = nn.Sequential(
                iresnet50(False, dropout=0, output_featmap=True),
                nn.Conv2d(64, ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ch, eps=1e-05),
                nn.PReLU(ch),
            )
            self.trans = nn.Sequential(
                T2T_ViT(
                    img_size=img_size // 2, tokens_type='face', in_chans=ch, num_classes=0,
                    embed_dim=embed_dim, depth=14, num_heads=6, mlp_ratio=3, drop_path_rate=0.1,
                    drop_rate=0.1),
            )
        else:
            ch = 3
            self.trans = nn.Sequential(
                T2T_ViT(
                    img_size=img_size, tokens_type='transformer', in_chans=ch, num_classes=0,
                    embed_dim=embed_dim, depth=14, num_heads=6, mlp_ratio=3, drop_path_rate=0.1,
                    drop_rate=0.1),
            )

        # self.layer = nn.Sequential(
        #        self.cnn,
        #        self.trans,
        #        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.BatchNorm1d(num_classes),
            # nn.BatchNorm1d(512, affine=False),
            # nn.BatchNorm1d(512),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, 0, 0.1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                # nn.init.normal(m.weight, std=.01)
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    # m.weight.requires_grad = False
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_cnn:
            x = self.cnn(x)
        x = self.trans(x)
        x = self.head(x)
        # x = self.backbone_cnn(x)
        # x = self.backbone_vit(x)
        return x
