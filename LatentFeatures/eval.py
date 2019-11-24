import torch.nn.functional as F
import torch

from utils import MCA

__all__ = ['evaluate']


def evaluate(model, valLoader, loss_func, unseen_class_attris, config, eval_type='ua'):
    """
    Args:
        eval_type (str, optional): evaluate type('ua', 'la', 'ua-la') default('ua')
    """
    assert eval_type in ('ua', 'la', 'ua-la')
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
    seens = 0
    for batch_idx, (imgs, targets) in enumerate(valLoader):
        imgs = imgs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            pred_attris = model(imgs)
            losses = [loss_func(attris[:, :config.TRAIN.num_attributes], attris[:, config.TRAIN.num_attributes:],
                                targets, unseen_class_attris) for attris in pred_attris]

            ua_loss = losses[0][0]
            la_loss = losses[0][-1]
            for loss_idx in range(1, len(losses)):
                ua_loss += losses[loss_idx][0]
                la_loss += losses[loss_idx][-1]
            loss = ua_loss + la_loss

            if eval_type == 'ua':
                pred_attris = pred_attris[0]
                _evaluate_UA(pred_attris[:, :config.TRAIN.num_attributes], unseen_class_attris, targets,
                             mean_class_accuracy)
            elif eval_type == 'la':
                raise NotImplementedError
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


def _evaluate_UA(pred_attributes, unseen_class_attributes, targets, MCA):
    pred = torch.mm(pred_attributes, unseen_class_attributes.t())
    pred = pred.max(dim=1)[1]
    MCA.update(targets.squeeze().cpu().numpy(), pred.cpu().numpy())


def _evaluate_LA():
    pass


def _evaluate_UALA():
    pass
