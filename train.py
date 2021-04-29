import os, time, sys
import argparse, logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from torchsummary import summary

import backbones
from backbones.vit_pytorch.t2t import T2TViT
from backbones.vit_pytorch.cvt import CvT
import losses
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler

torch.backends.cudnn.benchmark = True


def main(args):
    world_size = int(os.environ['WORLD_SIZE'])  # 总的gpu数量
    rank = int(os.environ['RANK'])  # gpu id
    dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    dropout = 0.4 if cfg.dataset == "webface" else 0
    # backbone = eval("backbones.{}".format(args.network))(False, dropout=dropout, fp16=cfg.fp16).to(local_rank)
    name_network = args.network
    if name_network == "r50":
        backbone = backbones.iresnet50(False, dropout=dropout).to(local_rank)
    elif name_network == "r100":
        backbone = backbones.iresnet100(False, dropout=dropout).to(local_rank)
    elif name_network == "t2t_vit":
        # backbone = backbones.t2t_vit.T2T_ViT(
        #     tokens_type='face', num_classes=512, embed_dim=768, depth=10, num_heads=6, mlp_ratio=3,
        #     drop_path_rate=0, drop_rate=0.1)
        backbone = T2TViT(image_size=112, num_classes=512, dim=384, depth=10, heads=6, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    elif name_network == "face_vit":
        backbone = backbones.face_vit.Face_ViT()
    elif name_network == "cvt":
        backbone = CvT(num_classes=512, s2_depth=4, s3_depth=16)
    elif name_network == "vit":
        backbone = backbones.vit_baseline()
    else:
        raise NotImplementedError("{} is not implemented!".format(name_network))
    summary(backbone.cuda(), input_size=(3, 112, 112), batch_size=-1)
    if args.resume:   # 继续训练
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    backbone.cuda()
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)   # 分布式广播, 从一个卡上广播到多个卡上.
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    margin_softmax = eval("losses.{}".format(args.loss))()
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    # opt_backbone = torch.optim.SGD(
    #     params=[{'params': backbone.parameters()}],
    #     lr=cfg.lr / 512 * cfg.batch_size * world_size, momentum=0.9, weight_decay=cfg.weight_decay)
    # opt_pfc = torch.optim.SGD(
    #     params=[{'params': module_partial_fc.parameters()}],
    #     lr=cfg.lr / 512 * cfg.batch_size * world_size, momentum=0.9, weight_decay=cfg.weight_decay)
    # scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    # scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    lr_mult1 = cfg.batch_size * world_size / 512.0 * 10
    # print('lr set:', cfg.opt1.lr)
    cfg.opt1.lr *= lr_mult1
    cfg.opt1.warmup_lr *= lr_mult1
    cfg.opt1.min_lr *= lr_mult1
    lr_mult2 = lr_mult1
    cfg.opt2.lr *= lr_mult2
    cfg.opt2.warmup_lr *= lr_mult2
    cfg.opt2.min_lr *= lr_mult2

    if cfg.opt1.opt == 'adamw':
        opt_backbone = torch.optim.AdamW(
            params=[{'params': backbone.parameters()}], lr=cfg.opt1.lr,
            weight_decay=cfg.opt1.weight_decay, eps=cfg.opt1.opt_eps)
    elif cfg.opt1.opt == 'sgd':
        opt_backbone = torch.optim.SGD(
            params=[{'params': backbone.parameters()}], lr=cfg.opt1.lr,
            weight_decay=cfg.opt1.weight_decay, momentum=cfg.opt1.momentum)
    else:
        raise NotImplementedError("OPT {} is not implemented!".format(cfg.opt1.opt))

    if cfg.opt2.opt == 'adamw':
        opt_pfc = torch.optim.AdamW(
            params=[{'params': module_partial_fc.parameters()}], lr=cfg.opt2.lr,
            weight_decay=cfg.opt2.weight_decay, eps=cfg.opt2.opt_eps)
    elif cfg.opt2.opt == 'sgd':
        opt_pfc = torch.optim.SGD(
            params=[{'params': module_partial_fc.parameters()}], lr=cfg.opt2.lr,
            weight_decay=cfg.opt2.weight_decay, momentum=cfg.opt2.momentum)
    else:
        raise NotImplementedError("OPT {} is not implemented!".format(cfg.opt2.opt))
    scheduler_backbone, _ = create_scheduler(cfg.opt1, opt_backbone)
    scheduler_pfc, _ = create_scheduler(cfg.opt2, opt_pfc)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info(f"num_epoch:{cfg.num_epoch}, total step is: {total_step}, "
                               f"step/epoch:{int(len(trainset) / cfg.batch_size/world_size)}")

    callback_verification = CallBackVerification(500, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(1000, rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16:
                features.backward(grad_scaler.scale(x_grad))
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            lr = opt_pfc.param_groups[0]["lr"]
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler, lr)
            callback_verification(global_step, backbone)
            callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step(epoch + 1)
        scheduler_pfc.step(epoch + 1)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='cvt', help='backbone network')  # r50 t2t_vit cvt
    parser.add_argument('--loss', type=str, default='CosFace', help='loss function')   # ArcFace
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)
