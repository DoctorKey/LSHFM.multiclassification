import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

from .. import ramps, cli
from ..eval import validate
from ..architectures import create_model, load_pretrained, fix_BN_stat, fix_BN_learn
from ..run_context import RunContext
from ..dataset.datasets import get_dataset_config
from ..dataset.dataloader import create_train_loader, create_eval_loader
from ..utils import save_checkpoint, AverageMeterSet, parameters_string, Prec1

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

def main(context):
    global global_step
    global best_prec1
    global args

    args = context.args

    context.vis_log.add_text('hparams', cli.arg2str(args))

    checkpoint_path = context.transient_dir

    start_time = time.time()
    dataset_config = get_dataset_config(args.dataset, args=args)
    train_dataset = dataset_config.get('train_dataset')
    val_dataset = dataset_config.get('val_dataset')
    num_classes = dataset_config.get('num_classes')
    train_loader = create_train_loader(train_dataset, args=args)
    if val_dataset is not None:
        eval_loader = create_eval_loader(val_dataset, args=args)
    else:
        eval_loader = None
    LOG.info("=> load dataset in {} seconds".format(time.time() - start_time))

    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    model = create_model(args.arch, num_classes, detach_para=False, DataParallel=False)
    LOG.info(parameters_string(model))

    model = load_pretrained(model, args.pretrained, args.arch, LOG)

    if args.class_criterion == 'CE':
        class_criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(args.class_criterion)

    if args.finetune_fc:
        paras = model.module.fc.parameters()
    else:
        paras = model.parameters()
    optimizer = torch.optim.SGD(paras, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.fix_BN_learn:
        model.apply(fix_BN_learn)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        validate(eval_loader, model, 0, context.vis_log, LOG, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, class_criterion, optimizer, epoch, context.vis_log)

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0 and eval_loader is not None:
            prec1 = validate(eval_loader, model, epoch, context.vis_log, LOG, args.print_freq)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1, LOG)

    save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, False, checkpoint_path, 'final', LOG)
    LOG.info("best_prec1 {}".format(best_prec1))


def train(train_loader, model, class_criterion, optimizer, epoch, writer):
    global global_step
    start_time = time.time()
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    if args.fix_BN_stat:
        model.apply(fix_BN_stat)

    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data[0], data[1]
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))   

        input, target = input.cuda(), target.cuda()
        
        model_out = model(input)
        if isinstance(model_out, tuple):
            feat, class_logit = model_out
        else:
            class_logit = model_out

        #output = class_logit
        class_loss = class_criterion(class_logit, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        global_step += 1

        meters.update('lr', optimizer.param_groups[0]['lr'])
        minibatch_size = len(target)
        meters.update('class_loss', class_loss.item())
        prec1 = Prec1(class_logit.data, target.data)
        meters.update('top1', prec1, minibatch_size)   

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'.format(
                    epoch, i, len(train_loader), meters=meters))

    LOG.info(' * TRAIN Prec@1 {top1.avg:.3f} ({0:.1f}/{1:.1f})'
          .format(meters['top1'].sum / 100, meters['top1'].count, top1=meters['top1']))
    if writer is not None:
        writer.add_scalar("train/lr", meters['lr'].avg, epoch)
        writer.add_scalar("train/class_loss", meters['class_loss'].avg, epoch)
        writer.add_scalar("train/prec1", meters['top1'].avg, epoch)
    LOG.info("--- training epoch in {} seconds ---".format(time.time() - start_time))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    if args.lr_rampup != 0:
        lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    if args.lr_reduce_epochs:
        reduce_epochs = [int(x) for x in args.lr_reduce_epochs.split(',')]
        for ep in reduce_epochs:
            if epoch >= ep:
                lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
