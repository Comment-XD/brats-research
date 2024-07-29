import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


class DiceCoefficient(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(DiceCoefficient, self).__init__()
        
        self.p = p # optional to increase the values of A and B by a power of "p"
        self.smooth = smooth # Prevents the denominator from becoming 0

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''

        # Approximation of |A U B|
        axes = tuple(torch.arange(len(labels.size()) - 1))
        probs = F.softmax(logits)

        # element-wise multiplication then summation
        
        numer = (probs * labels).sum(dim=axes)
        
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum(dim=axes)
        return torch.mean((2 * numer + self.smooth) / (denor + self.smooth))


def dice_loss(logits, labels, p=1, smooth=1):
    dice_coef = DiceCoefficient(p, smooth)
    return 1. - dice_coef(logits, labels)

        