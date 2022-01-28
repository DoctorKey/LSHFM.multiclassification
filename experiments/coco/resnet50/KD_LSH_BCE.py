import os
import sys
import logging

sys.path.append(".")
from src.multilabel import KD as main
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
    # Technical details
    #'gpus': '0,1,2,3',
    'gpus': '4,5,6,7',
    'omp_threads': 8,

    'workers': 8,
    'checkpoint_epochs': 20,
    'print-freq': 100,

    # Data
    'dataset': 'COCO2014',
    'resize_size': 512,
    'input_size': 448,

    # Data sampling
    'batch_size': 32,
    'eval_batch_size': 32,

    # Architecture
    'arch': 'resnet50',
    'fix_BN_stat': False,
    'fix_BN_learn': False,
    'force_2FC': True,
    #'pretrained': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'pretrained': 'pretrained/resnet50_feat_fc@resnet101_coco_l2_1.ckpt',
    'pretrained': 'pretrained/resnet50_feat_fc@resnet101_coco_lsh_1.ckpt',
    #'pretrained': 'pretrained/resnet50_feat_fc@resnet101_coco_lshl2_1.ckpt',

    'arch_t': 'resnet101',
    'pretrained_t': 'pretrained/resnet101_coco_BCE_77.67.ckpt',

    'class_criterion': 'BCE',

    'distill': 'lsh',
    'hash_num': 2048,
    #'std': 1,
    'bias': 'median',
    #'bias': '0',
    'LSH_loss': 'BCE',
    'beta': 3,

    'thre': 0.5,

    # Optimization
    'lr_rampup': 0,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'epochs': 60,
    'lr_rampdown_epochs': 60,
    #'lr_reduce_epochs': '20,40',

    
    #'pretrained': 'pretrained/ResNet18_KD_71.72_rmFC.ckpt',
    #'evaluate': True,
    #'pretrained': 'results/voc2007/resnet101/baseline_BCE/2020-11-06_08:56:49/transient/checkpoint.final.ckpt',
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
