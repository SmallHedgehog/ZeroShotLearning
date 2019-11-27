import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['L2Norm']


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    @staticmethod
    def L2(x, dim=-1):
        x = 1. * x / (torch.norm(x, dim=dim, keepdim=True).expand_as(x) + 1e-12)
        return x


class Normlize(nn.Module):
    def __init__(self):
        super(Normlize, self).__init__()

    @staticmethod
    def norm(x):
        # x = F.sigmoid(x)
        x = x - x.mean(dim=0)
        return L2Norm.L2(x)
