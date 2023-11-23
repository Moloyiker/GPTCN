import torch
from torch import nn
import torch.nn.functional as F


class masked_softmax_cross_entropy_loss(nn.Module):

    def __init__(self, weight=None):
        super(masked_softmax_cross_entropy_loss, self).__init__()
        self.register_buffer('weight', weight)

    def forward(self, input, target, mask):
        if not target.is_same_size(input):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()))

        input = F.softmax(input)
        loss = -torch.sum(target * torch.log(input + 1e-10), -1)
        # print(loss.shape)
        #loss = torch.unsqueeze(loss, 1)
        #mask /= torch.mean(mask)
        #mask = torch.unsqueeze(mask, 1)
        # print(mask.shape)
        mask = (~mask.bool()).int()
        loss = torch.mul(loss, mask)
        # print(loss)
        # print(torch.mean(loss))
        return torch.mean(loss)
