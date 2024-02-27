import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()

        self.n_kernels = n_kernels
        self.mul_factor = mul_factor
        self.bandwidth_multipliers = None
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):

        if self.bandwidth_multipliers is None or self.bandwidth_multipliers.device != X.device:
            self.bandwidth_multipliers = (self.mul_factor ** (torch.arange(self.n_kernels, device=X.device) - self.n_kernels // 2)).float()

        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances)

        bandwidth = bandwidth.to(X.device) if isinstance(bandwidth, torch.Tensor) else bandwidth
        exp_component = -L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers[:, None, None])
        return torch.exp(exp_component).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY