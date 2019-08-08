from ..registry import LOSSES
import torch.nn.functional as F

@LOSSES.register
def nll_loss(output, target):
    return F.nll_loss(output, target)