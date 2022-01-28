import torch
import torch.nn as nn


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, f_s, f_t):
        return self.loss(f_s, f_t)
