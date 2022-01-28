import os
import time
import logging

import torch
import torch.backends.cudnn as cudnn


from ..KD.LSH import LSH
from ..KD.L1 import L1
from ..KD.L2 import L2
from .. import ramps, cli
from ..eval import validate_coco, validate_voc, prec_recall_for_batch
from ..architectures import create_model, load_pretrained, fix_BN_stat, fix_BN_learn
from ..run_context import RunContext
from ..dataset.datasets import get_dataset_config
from ..dataset.dataloader import create_train_loader, create_eval_loader
from ..utils import save_checkpoint, AverageMeterSet, parameters_string

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

    dataset_config = get_dataset_config(args.dataset, args=args)
    train_dataset = dataset_config.get('train_dataset')
    val_dataset = dataset_config.get('val_dataset')
    num_classes = dataset_config.get('num_classes')
    train_loader = create_train_loader(train_dataset, args=args)
    if val_dataset is not None:
        eval_loader = create_eval_loader(val_dataset, args=args)
    else:
        eval_loader = None

    if 'voc' in args.dataset:
        validate = validate_voc
    else:
        validate = validate_coco

    LOG.info("=> creating teacher model '{}'".format(args.arch_t))
    model_t = create_model(args.arch_t, num_classes, detach_para=True, DataParallel=False)
    model_t = load_pretrained(model_t, args.pretrained_t, args.arch_t, LOG)
    model_t.eval()

    weight, bias = model_t.module.get_classifier_weight()
    t_dim = weight.shape[1]
    LOG.info('=> teacher feature dim: {}'.format(t_dim))
    LOG.info('=> teacher classifier weight std: {}'.format(weight.std()))
    if args.std is None:
        args.std = weight.std()
    if args.hash_num is None:
        args.hash_num = 32 * t_dim

    LOG.info("=> creating student model '{}'".format(args.arch))
    model = create_model(args.arch, num_classes, DataParallel=False, student_dim=t_dim, force_2FC=args.force_2FC)
    LOG.info(parameters_string(model))
    model = load_pretrained(model, args.pretrained, args.arch, LOG)

    LOG.info('=> creating {} for knowledge distillation'.format(args.distill))
    if args.distill == 'l1':
        criterion_kd = L1()
    elif args.distill == 'l2':
        criterion_kd = L2()
    elif args.distill == 'lsh':
        LOG.info('=> LSH: D:{} N:{} std:{} LSH_loss:{}'.format(t_dim, args.hash_num, args.std, args.LSH_loss))
        criterion_kd = LSH(t_dim, args.hash_num, args.std, with_l2=False, LSH_loss=args.LSH_loss)
    elif args.distill == 'lshl2':
        LOG.info('=> LSHl2: D:{} N:{} std:{} LSH_loss:{}'.format(t_dim, args.hash_num, args.std, args.LSH_loss))
        criterion_kd = LSH(t_dim, args.hash_num, args.std, with_l2=True, LSH_loss=args.LSH_loss)
    else:
        raise NotImplementedError(args.distill)

    criterion_kd = criterion_kd.cuda()

    # this stage only updates the feat_fc
    paras = model.module.feat_fc.parameters()

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

    if 'lsh' in args.distill:
        if args.bias == '0':
            LOG.info('=> init LSH bias by 0')
        elif args.bias == 'median':
            LOG.info('=> init LSH bias by median')
            criterion_kd.init_bias(model_t, train_loader, args.print_freq, use_median=True)
        elif args.bias == 'mean':
            LOG.info('=> init LSH bias by mean')
            criterion_kd.init_bias(model_t, train_loader, args.print_freq, use_median=False)
        else:
            raise NotImplementedError(args.bias)

    if args.evaluate:
        validate(eval_loader, model, args.thre, 0, context.vis_log, LOG, args.print_freq)
        return

    LOG.info('=> evaluate teacher')
    validate(eval_loader, model_t, args.thre, 0, context.vis_log, LOG, args.print_freq)

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, model, model_t, criterion_kd, optimizer, epoch, context.vis_log)

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': 0,
                'optimizer' : optimizer.state_dict(),
            }, False, checkpoint_path, epoch + 1, LOG)

    save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': 0,
            }, False, checkpoint_path, 'final', LOG)
    LOG.info("best_prec1 {}".format(best_prec1))


def train(train_loader, model, model_t, criterion_kd, optimizer, epoch, writer):
    global global_step
    start_time = time.time()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    if args.fix_BN_stat:
        model.apply(fix_BN_stat)
    model_t.eval()

    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data[0], data[1]
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))   

        input, target = input.cuda(), target.float().cuda()

        feat_s, logit_s = model(input, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        f_s = feat_s[-1]
        f_t = feat_t[-1]
        if args.distill == 'lshl2_s':
            loss_kd = criterion_kd(f_s, f_t, logit_t, target)
        else:
            loss_kd = criterion_kd(f_s, f_t)

        loss = args.beta * loss_kd

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        meters.update('lr', optimizer.param_groups[0]['lr'])
        minibatch_size = len(target)
        meters.update('loss', loss.item())
        meters.update('kd_loss', loss_kd.item())

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Loss {meters[loss]:.3f}\t'
                'KD {meters[kd_loss]:.4f}\t'
                .format(epoch, i, len(train_loader), meters=meters))


    if writer is not None:
        writer.add_scalar("train/lr", meters['lr'].avg, epoch)
        writer.add_scalar("train/loss", meters['loss'].avg, epoch)

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
