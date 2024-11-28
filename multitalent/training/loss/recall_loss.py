
import torch
import torch.nn as nn

from multitalent.training.loss.dice import SoftDiceLoss
from multitalent.training.loss.Recall import SoftRecallLoss
from multitalent.utilities.helpers import softmax_helper_dim1


class Recall_and_DC_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ignore_label=None):
        super().__init__()


        self.ignore_label = ignore_label

        self.recall_loss = SoftRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.dice_loss = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).float()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dice_loss(net_output, target_dice, loss_mask=mask)
        recall_loss = self.recall_loss(net_output, target_dice, loss_mask=mask)

        result = 0.25 * recall_loss + 0.75 * dc_loss
        return result