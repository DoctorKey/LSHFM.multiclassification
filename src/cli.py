import re
import argparse
import logging

from . import architectures
from .dataset import datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args', 'arg2str']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch Multi Labels')
    parser.add_argument('--dataset', metavar='DATASET', default='COCO2014',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: COCO2014)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('--arch-t', metavar='ARCH', default='resnet50',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--feat-fc-epoch', default=0, type=int, metavar='N',
                        help='number of train feat fc epochs')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-eb', '--eval-batch-size', default=256, type=int,
                        metavar='N', help='eval mini-batch size (default: 256)')
    parser.add_argument('--seed', default=0, type=int, metavar='N',
                        help='randomness seed')

    parser.add_argument('--omp-threads', default=4, type=int,
                        metavar='N', help="omp-threads")
    parser.add_argument('--gpus', type=str, default=None,
                        help='gpus')
    
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--lr-reduce-epochs', default=None, type=str, metavar='EPOCHS',
                        help='epochs when reduce learning rate')
    parser.add_argument('--lr-reduce-gamma', default=0.1, type=float,
                        metavar='LR', help='gamma for step reduce learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pre-trained model (default: none)')
    parser.add_argument('--pretrained-t', default='', type=str, metavar='PATH',
                        help='path to pre-trained teacher model (default: none)')
    parser.add_argument('--pretrained-t-init-FST', default='', type=str, metavar='PATH',
                        help='path to pre-trained teacher model (default: none)')

    parser.add_argument('--resize-size', default=256, type=int, help='resize-size')
    parser.add_argument('--input-size', default=224, type=int, help='input-size')

    parser.add_argument('--thre', default=0.5, type=float, metavar='N', help='threshold value')

    parser.add_argument('--fix-BN-stat', default=False, type=str2bool,
                        help='fix BN stat parameters', metavar='BOOL')
    parser.add_argument('--fix-BN-learn', default=False, type=str2bool,
                        help='fix BN learning parameters', metavar='BOOL')
    parser.add_argument('--finetune-fc', default=False, type=str2bool,
                        help='finetune only FC layer', metavar='BOOL')

    # Focal loss & ASL
    parser.add_argument('--class-criterion', type=str, default='BCE', choices=['CE', 'BCE'])
    parser.add_argument('--gamma-pos', type=float, default=0, help='focusing parameter')
    parser.add_argument('--gamma-neg', type=float, default=0, help='focusing parameter')
    parser.add_argument('--margin', type=float, default=0, help='margin for loss function')


    # LSH KD
    parser.add_argument('--distill', type=str, default='lshl2', choices=['kd', 'l2', 'lsh', 'lshl2'])
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('--beta', type=float, default=None, help='weight balance for other losses')
    parser.add_argument('--hash-num', default=None, type=int, help='the num of hash function (default: 32D)')
    parser.add_argument('--std', default=None, type=float, help='the std of LSH weight, default is the std of teacher fc weight')
    parser.add_argument('--bias', type=str, default='median', choices=['0', 'mean', 'median'])
    parser.add_argument('--LSH-loss', type=str, default='BCE', choices=['BCE', 'L1', 'L2'])
    parser.add_argument('--feat-dim', default=0, type=int, help='feature dimension')
    parser.add_argument('--force-2FC', default=False, type=str2bool, help='use 2FC in student', metavar='BOOL')

    parser.add_argument('--kd-T', type=float, default=4, help='temperature for KD distillation')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs

def arg2str(args):
    hparams_dict = vars(args)
    header = "| Key | Value |\n| :--- | :--- |\n"
    keys = hparams_dict.keys()
    #keys = sorted(keys)
    lines = ["| %s | %s |" % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + "\n".join(lines) + "\n"
    return hparams_table
