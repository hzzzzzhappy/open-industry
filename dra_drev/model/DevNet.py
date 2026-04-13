import torch
import torch.nn as nn
import torch.nn.functional as F


class DevNet(nn.Module):
    """1D conv scoring on patch features [B, C, N] -> scalar anomaly score per sample."""

    def __init__(self, args):
        super(DevNet, self).__init__()
        self.args = args
        self.in_c = 1152
        self.conv = nn.Conv1d(in_channels=self.in_c, out_channels=1, kernel_size=1, padding=0)

    def forward(self, feature, label=None):
        if feature.ndim == 3 and feature.shape[2] == self.in_c:
             feature = feature.permute(0, 2, 1)

        image_pyramid = list()

        scores = self.conv(feature)

        if self.args.topk > 0:
            scores = scores.view(int(scores.size(0)), -1)
            topk = max(int(scores.size(1) * self.args.topk), 1)
            scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
            scores = torch.mean(scores, dim=1).view(-1, 1)
        else:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.mean(scores, dim=1).view(-1, 1)

        image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)

        return [score.view(-1, 1)]
