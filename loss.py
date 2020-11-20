import torch.nn as nn
import torch
import numpy as np




def _fast_hist(true, pred, num_classes):
    pred = np.round(pred).astype(int)
    true = np.round(true).astype(int)
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(np.float)
    return hist

def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = np.diag(hist)
    A = np.sum(hist,axis=1)
    B = np.sum(hist,axis=0)
    jaccard = A_inter_B / (A + B - A_inter_B + 1e-6)
    avg_jacc =np.nanmean(jaccard) #the mean of jaccard without NaNs
    return avg_jacc, jaccard

def dice_coef_metric(hist):
    """Computes the dice coefficient).
    Args:
        hist: confusion matrix.
     Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = np.diag(hist)
    A = np.sum(hist,axis=1)
    B = np.sum(hist,axis=0)
    dsc = A_inter_B * 2 / (A + B + 1e-6)
    avg_dsc=np.nanmean(dsc) #the mean of dsc without NaNs
    return avg_dsc


def dice_coef_loss(y_pred, y_true):
      smooth=1.0
      assert y_pred.size() == y_true.size()
      intersection = (y_pred * y_true).sum()
      dsc = (2. * intersection + smooth) / (
          y_pred.sum() + y_true.sum() + smooth
      )
      return 1. - dsc


def bce_dice_loss(y_pred, y_true):
    dicescore = dice_coef_loss(y_pred, y_true)
    bcescore = nn.BCELoss()
    m = nn.Sigmoid()
    bceloss = bcescore(m(y_pred), y_true)
    return (bceloss + dicescore)
