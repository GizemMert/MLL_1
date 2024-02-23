import torch
import torch.nn as nn

def make_positive_definite(cov_matrix):
    regularization_term = 1e-6
    identity_matrix = torch.eye(cov_matrix.size(0), dtype=cov_matrix.dtype, device=cov_matrix.device)
    return cov_matrix + regularization_term * identity_matrix


def mmd(source, target):
    mseloss = nn.MSELoss()
    kldiv = nn.KLDivLoss()

    source_mu = torch.mean(source, axis=0)
    target_mu = torch.mean(target, axis=0)
    mu_dis = mseloss(source_mu, target_mu)

    source_cov = torch.cov(source.T)
    target_cov = torch.cov(target.T)

    source_cov = make_positive_definite(source_cov)
    target_cov = make_positive_definite(target_cov)

    cov_loss = 0.5 * (
            kldiv(torch.nn.LogSoftmax(dim=0)(source_cov), torch.nn.Softmax(dim=0)(target_cov)) +
            kldiv(torch.nn.LogSoftmax(dim=0)(target_cov), torch.nn.Softmax(dim=0)(source_cov)))

    loss = mu_dis + cov_loss
    return loss