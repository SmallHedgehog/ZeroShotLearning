import argparse
import yaml
import os
import os.path as osp
import time
import datetime
import random
import numpy as np
import torch
import torchvision.transforms as trans
import _init_path

from terminaltables import AsciiTable
from easydict import EasyDict
from dataset import CUB_200_2011
from utils import Logger
from models import LFNet, L2Norm, Normlize
from loss import SoftmaxLoss, TripletLoss
from eval import evaluate

GPU_ID = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


def parser_args():
    parser = argparse.ArgumentParser('(FG)ZeroShotLearning-Latent features learning')
    parser.add_argument('--config', type=str, default='../experiments/SS_AE_Learned(VGG19)_NORM/config.yaml')
    args = parser.parse_args()
    with open(args.config) as file:
        config = EasyDict(yaml.safe_load(file))
    return config, args


if __name__ == '__main__':
    config, args = parser_args()
    print(config)

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logger = Logger(osp.join(osp.dirname(args.config), 'logs'))
    os.makedirs(config.CHECKPOINTS_PATH, exist_ok=True)

    # Augment
    train_transform = trans.Compose([
        trans.RandomResizedCrop(224),
        # trans.Resize((224, 224)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()])
    val_transform = trans.Compose([
        trans.Resize(256),
        trans.CenterCrop(224),
        # trans.Resize((224, 224)),
        trans.ToTensor()])

    # Get dataloader
    trainSet = CUB_200_2011(config.DATA_PATH, phase='train', transform=train_transform)
    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=config.TRAIN.batch_size,
        shuffle=True,
        num_workers=config.TRAIN.num_workers,
        pin_memory=True)
    valSet = CUB_200_2011(config.DATA_PATH, phase='val', transform=val_transform)
    valLoader = torch.utils.data.DataLoader(
        valSet,
        batch_size=config.TRAIN.batch_size,
        shuffle=True,
        num_workers=config.TRAIN.num_workers,
        pin_memory=True)

    # Initiate model
    model = LFNet(
        num_attributes=config.TRAIN.num_attributes,
        augmented=config.TRAIN.augmented,
        num_scales=config.TRAIN.num_scales,
        backbone=config.TRAIN.backbone,
        pretrained=config.TRAIN.pretrained)
    model = model.cuda()
    seen_class_attributes = L2Norm.L2(trainSet.get_class_attributes.cuda())
    unseen_class_attributes = L2Norm.L2(valSet.get_class_attributes.cuda())

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.TRAIN.lr),
        momentum=config.TRAIN.momentum,
        weight_decay=float(config.TRAIN.weight_decay))
    # optimizer = torch.optim.Adam(model.parameters(), lr=float(config.TRAIN.lr))

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.TRAIN.max_epochs
    )

    # Softmax loss
    softmax_loss = SoftmaxLoss()
    # Triplet loss
    triplet_loss = TripletLoss(config.TRAIN.margin)

    if config.TRAIN.augmented:
        loss_func = lambda x1, x2, y, class_attris: (softmax_loss(x1, class_attris, y), triplet_loss(x2, y))
    else:
        loss_func = lambda x1, x2, y, class_attris: (softmax_loss(x1, class_attris, y), )

    # Metrics
    metrics = [
        'loss'
    ]
    BEST_accuracy = 0.
    for epoch in range(config.TRAIN.max_epochs):
        model.train()
        start_time = time.time()
        for batch_idx, (imgs, targets) in enumerate(trainLoader):
            batches_done = len(trainLoader) * epoch + batch_idx
            imgs = imgs.cuda()
            targets = targets.cuda()

            pred_attris = model(imgs)
            losses = [loss_func(attris[:, :config.TRAIN.num_attributes], attris[:, config.TRAIN.num_attributes:],
                                targets, seen_class_attributes) for attris in pred_attris]

            ua_loss = losses[0][ 0]
            la_loss = losses[0][-1]
            for idx in range(1, len(losses)):
                ua_loss += losses[idx][ 0]
                la_loss += losses[idx][-1]

            optimizer.zero_grad()
            if config.TRAIN.augmented:
                loss = ua_loss + la_loss
            else:
                loss = ua_loss
            loss.backward()
            optimizer.step()

            log_str = '\n---- [Epoch {}/{}, Batch {}/{}] ----\n'.format(
                epoch, config.TRAIN.max_epochs, batch_idx, len(trainLoader))
            metric_table = [['Metrics', 'User attributes', 'Latent attributes', 'Total']]
            for idx, metric in enumerate(metrics):
                row_metrics = ["%.4f" % v for v in [ua_loss.item(), la_loss.item(), loss.item()]]
                metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            tensorboard_log  = [('UA_Loss', ua_loss.item())]
            tensorboard_log += [('LA_Loss', la_loss.item())]
            tensorboard_log += [('Total_Loss', loss.item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            # Determine approximate time left for epoch
            epoch_batches_left = len(trainLoader) - (batch_idx + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_idx + 1))
            log_str += '\n---- ETA {}'.format(time_left)

            print(log_str)

        # Evaluation
        val_loss, val_accuracy = evaluate(
            model=model,
            valLoader=valLoader,
            loss_func=loss_func,
            unseen_class_attris=unseen_class_attributes,
            config=config,
            eval_type='ua'
        )

        logger.list_of_scalars_summary([(key, val_loss[key]) for key in val_loss.keys()], epoch, 'val_loss')
        logger.list_of_scalars_summary([(key, val_accuracy[key]) for key in val_accuracy], epoch, 'val_accuracy')

        print('\n---- Evaluating Model ----')
        metric_table = [['Metrics', 'User attributes', 'Latent attributes', 'Total']]
        metric_table += [['loss', val_loss['User attributes'], val_loss['Latent attributes'], val_loss['Total']]]
        metric_table += [['MCA', val_accuracy['User attributes'], val_accuracy['Latent attributes'],
                          val_accuracy['Total']]]
        print(AsciiTable(metric_table).table)

        if val_accuracy['Total'] >= BEST_accuracy:
            BEST_accuracy = val_accuracy['Total']
            infos = {
                'MODEL': model.state_dict(),
                'ACCURACY': BEST_accuracy
            }
            torch.save(infos, osp.join(config.CHECKPOINTS_PATH, 'LFNet.pth'))

        scheduler.step()
