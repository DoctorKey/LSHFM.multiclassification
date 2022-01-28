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
    'pretrained': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'arch_t': 'resnet34',
    'pretrained_t': 'pretrained/resnet34_voc2007_BCE_91.69.ckpt',

    'class_criterion': 'BCE',
    'gamma': 0,

    'distill': 'l2',
    'beta': 1,
    'force_2FC': True,

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

    'finetune_fc': True,

    
    #'evaluate': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
