import torch.nn as nn
import torch

__all__ = ['L2Norm']


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    @staticmethod
    def L2(x):
        l2_norm = torch.norm(x, dim=1)
        l2_norm = l2_norm.unsqueeze(1)
        return x / l2_norm
