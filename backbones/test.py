from backbones import t2t_vit
from mmcv.cnn import get_model_complexity_info
import torch

if __name__ == '__main__':
    model = t2t_vit.T2T_ViT(
        tokens_type='face', in_chans=3, num_classes=512,
        embed_dim=512, depth=10, num_heads=8, mlp_ratio=3,
        drop_path_rate=0, drop_rate=0.1)
    model(torch.ones([2, 3, 112, 112]))

    flops, params = get_model_complexity_info(model, (3, 112, 112))
    print(flops)
    print(params)