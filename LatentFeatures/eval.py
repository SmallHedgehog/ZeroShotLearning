import torch.nn.functional as F
import torch

from sklearn.linear_model import Ridge

from utils import MCA

__all__ = ['evaluate']


def evaluate(model, trainLoader, valLoader, loss_func, seen_class_attris, unseen_class_attris, config):
    """
    Args:
        eval_type (str, optional): evaluate type('ua', 'la', 'ua-la') default('ua')
    """
    assert config.VAL.evaluate in ('ua', 'la', 'ua-la')
    model.eval()
    mean_class_accuracy = MCA(config.TRAIN.unseens)
    loss_metrics = {
        'User attributes': 0.,
        'Latent attributes': 0.,
        'Total': 0.
    }
    mca_metrics = {
        'User attributes': 0.,
        'Latent attributes': 0.,
        'Total': 0.
    }

    if 'la' in config.VAL.evaluate:
        seen_class_latent_attris = _get_seen_classes_latent_attris(
            model, trainLoader, seen_class_attris.size(0), seen_class_attris.size(1))
        unseen_class_latent_attris = _get_unseen_classes_latent_attris(
            seen_class_attris, unseen_class_attris, seen_class_latent_attris)

    seens = 0
    for batch_idx, (imgs, targets) in enumerate(valLoader):
        imgs = imgs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            pred_attris = model(imgs)
            losses = [loss_func(attris[:, :config.TRAIN.num_attributes], attris[:, config.TRAIN.num_attributes:],
                                targets, unseen_class_attris) for attris in pred_attris]

            ua_loss = losses[0][ 0]
            la_loss = losses[0][-1]
            for loss_idx in range(1, len(losses)):
                ua_loss += losses[loss_idx][ 0]
                la_loss += losses[loss_idx][-1]
            loss = ua_loss + la_loss

            if config.VAL.evaluate == 'ua':
                pred_attris = pred_attris[0]    # Note that this implement only one scale.
                _evaluate(pred_attris[:, :config.TRAIN.num_attributes], unseen_class_attris, targets,
                          mean_class_accuracy)
            elif config.VAL.evaluate == 'la':
                pred_attris = pred_attris[0]    # Note that this implement only one scale.
                _evaluate(pred_attris[:, config.TRAIN.num_attributes:], unseen_class_latent_attris, targets,
                          mean_class_accuracy)
            else:
                raise NotImplementedError

            seens += imgs.size(0)
            loss_metrics['User attributes'] += ua_loss.item()
            loss_metrics['Latent attributes'] += la_loss.item()
            loss_metrics['Total'] += loss.item()

    loss_metrics['User attributes'] /= len(valLoader)
    loss_metrics['Latent attributes'] /= len(valLoader)
    loss_metrics['Total'] /= len(valLoader)

    mca = mean_class_accuracy.calc()
    mca_metrics['User attributes'] = mca
    mca_metrics['Latent attributes'] = mca
    mca_metrics['Total'] = mca

    return loss_metrics, mca_metrics


def _evaluate(pred_attris, unseen_class_attris, targets, MCA):
    pred = torch.mm(pred_attris, unseen_class_attris.t())
    pred = pred.max(dim=1)[1]
    MCA.update(targets.squeeze().cpu().numpy(), pred.cpu().numpy())


def _evaluate_UALA():
    pass


def _get_seen_classes_latent_attris(model, trainLoader, num_classes, num_attris):
    latent_attris = torch.zeros((num_classes, num_attris), dtype=torch.float).cuda()
    nums = torch.zeros(num_classes, dtype=torch.float)
    for _, (imgs, targets) in enumerate(trainLoader):
        imgs = imgs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            pred_attris = model(imgs)[0][:, :num_attris]    # Note that this implement only one scale.
            for class_idx in range(num_classes):
                indxs = (targets == class_idx)
                nums[class_idx] += indxs.sum()
                latent_attris[class_idx] += pred_attris[indxs].sum(dim=0)

    return latent_attris / nums.unsqueeze(1).cuda()


def _get_unseen_classes_latent_attris(seen_class_attris, unseen_class_attris, seen_class_latent_attris, gamma=1.0):
    ridge = Ridge(alpha=gamma, fit_intercept=False)
    ridge.fit(seen_class_attris.cpu().t().numpy(), unseen_class_attris.cpu().t().numpy())
    betas = torch.from_numpy(ridge.coef_).cuda()
    return torch.mm(betas, seen_class_latent_attris)
