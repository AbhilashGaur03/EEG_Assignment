"""Paper-faithful EEGNet implementation."""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet-8,2 PyTorch-style implementation."""

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 321,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kern_len: int = 80,
        dropout: float = 0.5,
    ):
        super().__init__()
        F2 = F1 * D
        self.F2 = F2

        self.temporal = nn.Conv2d(1, F1, kernel_size=(1, kern_len), padding=(0, kern_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depth = nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False)
        self.sep_point = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            h = self.features(dummy)
            self.flat_dim = h.shape[1]

        self.fc = nn.Linear(self.flat_dim, n_classes)

    def features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def extract_features(self, x):
        return self.features(x)

    def forward(self, x, return_features=False):
        h = self.features(x)
        logits = self.fc(h)
        if return_features:
            return logits, h
        return logits


def project_eegnet_max_norm(model, depthwise_max_norm=1.0, dense_max_norm=0.25):
    """Apply EEGNet max-norm constraints after optimizer.step()."""

    def project_weight(weight, max_norm):
        with torch.no_grad():
            w = weight.data
            w_flat = w.reshape(w.shape[0], -1)
            norms = w_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norms, max=max_norm)
            w_flat.mul_(desired / norms)

    if hasattr(model, "depthwise"):
        project_weight(model.depthwise.weight, depthwise_max_norm)
    if hasattr(model, "fc"):
        project_weight(model.fc.weight, dense_max_norm)
