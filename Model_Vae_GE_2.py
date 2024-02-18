import torch.nn as nn
import torch
from torch import Tensor
from torch._inductor.ir import View
from torch.autograd import Variable

class VAE_GE(nn.Module):
    def __init__(self, input_shape=None, latent_dim=50):
        super(VAE_GE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, 20000),
            nn.BatchNorm1d(20000),
            nn.ReLU(),

            nn.Linear(20000, 8000),
            nn.BatchNorm1d(8000),
            nn.ReLU(),



            nn.Linear(8000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),

            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 2 * self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 500),
            nn.ReLU(),


            nn.Linear(500, 1000),
            nn.ReLU(),

            nn.Linear(1000, 2000),
            nn.ReLU(),

            nn.Linear(2000, 8000),
            nn.ReLU(),


            nn.Linear(8000, 20000),
            nn.ReLU(),


            nn.Linear(20000, self.input_shape),
            nn.ReLU()

        )



    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        y = self.decoder(z)

        return z, y, mu, logvar

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def reparametrize(mu, log_var):
    std = log_var.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps