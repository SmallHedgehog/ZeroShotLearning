import torch.nn as nn
import torch


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, class_attributes, y):
        """
        Args:
            x (batch_size, num_attributes)
            class_attributes (num_classes, num_attributes)
            y (batch_size, 1)
        """
        pred = torch.mm(x, class_attributes.t())
        return self.cross_entropy(pred, y)
