import os
import sys
import logging

sys.path.append(".")
from src.multilabel import KD as main
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
        # Technical details
        #'gpus': '0,1',
        #'gpus': '4,5',
        #'gpus': '2,3',
        'gpus': '0,1,2,3',
        #'gpus': '4,5,6,7',
        'omp_threads': 8,

        'workers': 8,
        'checkpoint_epochs': 80,
        'print-freq': 50,

        # Data
        'dataset': 'voc2007',
        'resize_size': 512,
        'input_size': 448,

        # Data sampling
        'batch_size': 16,
        'eval_batch_size': 64,

        # Architecture
        'arch': 'resnet18',
        'force_2FC': True,
        #'pretrained': 'pretrained/r18_feat_fc@r34_voc_L2_1.ckpt',
        #'pretrained': 'pretrained/r18_feat_fc@r34_voc_lsh_1.ckpt',
        'pretrained': 'pretrained/r18_feat_fc@r34_voc_lshl2_1.ckpt',

        'arch_t': 'resnet34',
        'pretrained_t': 'pretrained/resnet34_voc2007_BCE_91.69.ckpt',

        'class_criterion': 'BCE',

        'distill': 'lsh',
        'hash_num': 2048,
        #'std': 1,
        'bias': 'median',
        #'bias': '0',
        'LSH_loss': 'BCE',
        'beta': 0.5,


        'thre': 0.5,

        # Optimization
        'lr_rampup': 0,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'momentum': 0.9,
        'nesterov': False,
        'epochs': 60,
        'lr_rampdown_epochs': 60,
        #'lr_reduce_epochs': '20,40',

        #'pretrained': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        #'pretrained': 'pretrained/ResNet18_KD_71.72_rmFC.ckpt'
        #'evaluate': True,
    }


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
