from datetime import datetime
import logging
import os
import socket

import torch
import numpy as np
import random

from .cli import parse_dict_args

class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, parameters_dict, log=True, tensorboard=False):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        dataset = runner_file.split('/')[1]
        network = runner_file.split('/')[2]
        runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = "{root}/{dataset}/{network}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}".format(
            root='results',
            dataset=dataset,
            network=network,
            runner_name=runner_name,
            date=datetime.now(),
        )
        self.transient_dir = self.result_dir + "/transient" 
        if log:  
            os.makedirs(self.result_dir)
            os.makedirs(self.transient_dir)

        self._init_log(log)
        self.args = parse_dict_args(**parameters_dict)
        self._init_env()
        self.vis_log = Tensorboard_logger(self.result_dir + "/TB_log", not tensorboard)
        set_random_seed(self.args.seed)

    def _init_env(self):
        os.environ['OMP_NUM_THREADS'] = str(self.args.omp_threads)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpus

    def _init_log(self, log):
        LOG = logging.getLogger('main')
        FileHandler = logging.FileHandler(os.path.join(self.result_dir, 'log.txt'))
        LOG.addHandler(FileHandler)



class Tensorboard_logger(object):
    """docstring for Tensorboard_logger"""
    def __init__(self, save_dir, isNone=False):
        super(Tensorboard_logger, self).__init__()
        self.save_dir = save_dir
        hostname = socket.gethostname()
        if 'GPU2' in hostname or 'gpu1' in hostname or isNone:
            self.logger = None
        else:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(save_dir)

    def add_scalar(self, name, value, step):
        if self.logger:
            self.logger.add_scalar(name, value, step)

    def add_text(self, name, value):
        if self.logger:
            self.logger.add_text(name, value)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)