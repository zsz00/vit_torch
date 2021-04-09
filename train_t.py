import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.nn as nn
from torch.optim import AdamW
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import backbones
import losses
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

torch.backends.cudnn.benchmark = True


def main(args):
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print('run with args:', args)
        print('run with cfg:', cfg)

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    dropout = 0.4 if cfg.dataset == "webface" else 0
    # name_network = "transform"
    # name_network = "cnn_transform"
    name_network = args.network
    if name_network == "r50":
        backbone = backbones.iresnet50(False, dropout=dropout).to(local_rank)
    elif name_network == "r100":
        backbone = backbones.iresnet100(False, dropout=dropout).to(local_rank)
    elif name_network == "transform":
        backbone = backbones.t2t_vit.T2T_ViT(
            tokens_type='face', in_chans=3, num_classes=512,
            embed_dim=768, depth=20, num_heads=6, mlp_ratio=3,
            drop_path_rate=0, drop_rate=0.1)
    elif name_network == "cnn_transform":
        backbone = backbones.face_vit.Face_ViT()
    elif name_network == "vit":
        backbone = backbones.vit_baseline()
    else:
        raise NotImplementedError("{} is not implemented!".format(name_network))

    # try:
    #    backbone.load_state_dict(torch.load(os.path.join(cfg.output, "backbone.pth"),
    #                                        map_location=torch.device(local_rank)))
    #    if rank == 0:
    #        logging.info("backbone resume successfully!")
    # except (FileNotFoundError, KeyError, IndexError, RuntimeError):
    #    logging.info("No pretrain found, backbone init successfully!")

    # for ps in backbone.parameters():
    #     dist.broadcast(ps, 0)

    name_loss = "cosface"
    if name_loss == "arcface":
        margin_softmax = losses.ArcFace(s=64.0, m=0.5)
    elif name_loss == "cosface":
        margin_softmax = losses.CosFace(s=64.0, m=0.4)
    else:
        raise NotImplementedError("{} is not implemented!".format(name_loss))

    backbone.cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone_without_ddp = backbone.module
    backbone.train()

    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size,
        batch_size=cfg.batch_size, resume=args.resume, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        embedding_size=cfg.embedding_size, prefix=cfg.output)

    # opt_backbone = AdamW(
    #    params=[{'params': backbone.parameters()}], lr=cfg.lr / 512 * cfg.batch_size * world_size,
    #    weight_decay=0.05, rescale=1, eps=cfg.opt_eps)
    # opt_pfc = AdamW(
    #    params=[{'params': module_partial_fc.parameters()}], lr=cfg.lr / 512 * cfg.batch_size * world_size,
    #    weight_decay=0.0005, rescale=world_size, eps=cfg.opt_eps)

    lr_mult1 = cfg.batch_size * world_size / 512.0 * 10
    # print('lr set:', cfg.opt1.lr)
    cfg.opt1.lr *= lr_mult1
    cfg.opt1.warmup_lr *= lr_mult1
    cfg.opt1.min_lr *= lr_mult1

    # lr_mult2 = cfg.batch_size / 512.0
    lr_mult2 = lr_mult1
    cfg.opt2.lr *= lr_mult2
    cfg.opt2.warmup_lr *= lr_mult2
    cfg.opt2.min_lr *= lr_mult2
    # print('lr used:', cfg.opt1.lr)
    # pfc_rescale = 1.0 / world_size
    # pfc_rescale = 1.0
    # cfg.lr *= pfc_rescale
    # cfg.warmup_lr *= pfc_rescale
    # cfg.min_lr *= pfc_rescale

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

    # opt_backbone = torch.optim.AdamW(
    #    params=[{'params': backbone.parameters()}], lr=cfg.lr,
    #    weight_decay=0.05, eps=cfg.opt_eps)
    # scheduler_backbone, _ = create_scheduler(cfg, opt_backbone)
    # opt_pfc = torch.optim.AdamW(
    #    params=[{'params': module_partial_fc.parameters()}], lr=cfg.lr,
    #    weight_decay=0.05, eps=cfg.opt_eps)
    # scheduler_pfc, _ = create_scheduler(cfg, opt_pfc)
    # scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
    #    optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    # scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
    #    optimizer=opt_pfc, lr_lambda=cfg.lr_func)
    start_epoch = 0

    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.opt1.epochs)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(4000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, cfg.opt1.epochs):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, None)
            features.backward(x_grad)
            nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_pfc.step()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)

            callback_logging(global_step, loss, epoch, opt_backbone, opt_pfc)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step(epoch + 1)
        scheduler_pfc.step(epoch + 1)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='r50', help='')   # transform
    parser.add_argument('--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args = parser.parse_args()
    main(args)
