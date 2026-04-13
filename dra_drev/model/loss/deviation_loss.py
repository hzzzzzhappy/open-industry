import torch
import torch.nn as nn

class DeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        confidence_margin = 5.
        # create reference noise on the same device and dtype as y_pred
        device = y_pred.device if isinstance(y_pred, torch.Tensor) else torch.device('cpu')
        dtype = y_pred.dtype if isinstance(y_pred, torch.Tensor) else torch.float32
        ref = torch.randn(5000, device=device, dtype=dtype)
        # ref = torch.normal(mean=0., std=torch.full([5000], 1., device=device, dtype=dtype), device=device, dtype=dtype)
        # ensure y_true is on same device
        if isinstance(y_true, torch.Tensor) and y_true.device != device:
            y_true = y_true.to(device)

        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        dev_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        return torch.mean(dev_loss)
