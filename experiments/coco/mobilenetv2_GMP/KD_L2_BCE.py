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
    'checkpoint_epochs': 20,
    'print-freq': 200,

    # Data
    'dataset': 'COCO2014',
    'input_size': 448,

    # Data sampling
    'batch_size': 32,
    'eval_batch_size': 32,

    # Architecture
    'arch': 'mobilenet_v2_GMP',
    'fix_BN_stat': False,
    'fix_BN_learn': False,
    'force_2FC': True,
    #'pretrained': 'pretrained/mnv2_feat_fc@resnet101_coco_l2_1.ckpt',
    #'pretrained': 'pretrained/mnv2_feat_fc@resnet101_coco_lsh_1.ckpt',
    'pretrained': 'pretrained/mnv2_feat_fc@resnet101_coco_lshl2_1.ckpt',

    'arch_t': 'resnet101_GMP',
    'pretrained_t': 'pretrained/resnet101_GMP_coco_BCE_79.57.ckpt',

    'class_criterion': 'BCE',

    'distill': 'l2',
    'beta': 3,

    # Optimization
    'lr_rampup': 0,
    'lr': 0.01,
    'weight_decay': 1e-4,
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
