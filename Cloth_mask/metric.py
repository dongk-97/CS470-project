
import torch

eps = 1e-8
threshold = 0.5


def get_similiarity(output, target, ch=None):
    """
    Call to calculate similiarity metrics for train, validation and test
    @ inputs:
        output : predicted model result (N x C x H x W)
        target : one-hot formatted labels (N x C x H x W)
    @ outputs:
        dice similiarity = 2 * inter / (inter + union + e)
        jaccard similiarity = inter / (union + e)
    """
    if ch is not None:
        output1 = output[:, ch, :, :]
        target1 = target[:, ch, :, :]
    else:
        output1 = output
        target1 = target

    intersection = torch.sum(output1 * target1.float())
    union = torch.sum(output1) + torch.sum(target1.float()) - intersection
    dice = 2 * intersection / (union + intersection + eps)
    # jaccard = intersection / (union + eps)
    return dice  #, jaccard
