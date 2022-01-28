import os
import sys
import logging

sys.path.append(".")
from src.multilabel import KD as main
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
    # Technical details
    'gpus': '0,1,2,3',
    #'gpus': '4,5,6,7',
    'omp_threads': 8,

    'workers': 8,
    'checkpoint_epochs': 80,
    'print-freq': 50,

    # Data
    'dataset': 'voc2007',
    #'resize_size': 512,
    'input_size': 448,

    # Data sampling
    'batch_size': 16,
    'eval_batch_size': 64,

    # Architecture
    'arch': 'resnet50',
    #'pretrained': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'pretrained': 'pretrained/ResNet50_KD@ResNet101_77.57_rmFC.ckpt',
    #'force_2FC': True,
    #'pretrained': 'pretrained/r50_feat_fc@r101_voc_l2_1.ckpt',
    #'pretrained': 'pretrained/r50_feat_fc@r101_voc_lshl2_1.ckpt',

    'arch_t': 'resnet101',
    'pretrained_t': 'pretrained/resnet101_voc2007_BCE_93.27.ckpt',

    'class_criterion': 'BCE',

    'distill': 'lshl2',
    'hash_num': 8192,
    #'std': None,
    'bias': 'median',
    'LSH_loss': 'BCE',
    'beta': 0.5,


    # Optimization
    'lr_rampup': 0,
    'lr': 0.01,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'nesterov': False,
    'epochs': 60,
    'lr_rampdown_epochs': 60,
    #'lr_reduce_epochs': '20,40',

    'thre': 0.5,
    #'evaluate': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
