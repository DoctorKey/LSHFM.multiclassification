import torch
import torch.nn as nn


class MultiCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiCrossEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input, target):
        logit = self.logsoftmax(input)
        loss = - logit * target
        loss = loss.sum() / target.sum()
        return loss

