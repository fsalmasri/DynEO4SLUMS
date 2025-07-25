import torch
import kornia
import torch.nn as nn
from torch.nn import functional as F

from basicsr.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        ssim_loss = kornia.losses.ssim_loss(x, gt, window_size=5, reduction="none")
        ssim_loss = torch.mean(ssim_loss.mean(dim=(-1,-2,-3)))
        return ssim_loss * self.loss_weight
