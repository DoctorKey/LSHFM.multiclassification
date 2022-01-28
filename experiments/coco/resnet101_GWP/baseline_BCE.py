import os
import sys
import logging

sys.path.append(".")
from src.multilabel import baseline as main
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
    'input_size': 448,

    # Data sampling
    'batch_size': 32,
    'eval_batch_size': 32,

    # Architecture
    'arch': 'resnet101_GWP',

    'class_criterion': 'BCE',

    # Optimization
    'lr_rampup': 0,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'epochs': 30,
    'lr_rampdown_epochs': 30,
    #'lr_reduce_epochs': '20,40',

    'pretrained': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'evaluate': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
