import torch.optim as optimizer

__all__ = ['get_optimizer']


def get_optimizer(model, config):
    if config.TRAIN.optimizer == 'sgd':
        return optimizer.SGD(model.parameters(), lr=float(config.TRAIN.lr), momentum=float(config.TRAIN.momentum),
                             weight_decay=float(config.TRAIN.weight_decay))
    elif config.TRAIN.optimizer == 'adam':
        return optimizer.Adam(model.parameters(), lr=float(config.TRAIN.lr))
    else:
        raise ValueError
