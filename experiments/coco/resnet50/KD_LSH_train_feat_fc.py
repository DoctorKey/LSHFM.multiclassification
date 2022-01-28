import os
import sys
import logging

sys.path.append(".")
from src.multilabel import KD_train_feat_fc as main
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
    # Technical details
    #'gpus': '0,1,2,3',
    'gpus': '4,5,6,7',
    #'gpus': '12,13,14,15',
    'omp_threads': 8,

    'workers': 8,
    'checkpoint_epochs': 80,
    'print-freq': 200,

    # Data
    'dataset': 'COCO2014',
    'resize_size': 512,
    'input_size': 448,

    # Data sampling
    'batch_size': 32,
    'eval_batch_size': 32,

    # Architecture
    'arch': 'resnet50',

    'pretrained': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'pretrained': 'pretrained/ResNet18_KD_71.72_rmFC.ckpt',

    'arch_t': 'resnet101',
    'pretrained_t': 'pretrained/resnet101_coco_BCE_77.67.ckpt',

    'class_criterion': 'BCE',
    'gamma': 0,

    'distill': 'lsh',
    'beta': 1,
    'force_2FC': True,

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

    'finetune_fc': True,
    #'evaluate': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
