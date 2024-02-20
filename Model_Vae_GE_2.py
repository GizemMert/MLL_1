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
            nn.Linear(self.input_shape, 800),
            # nn.BatchNorm1d(1500),
            nn.LeakyReLU(),

            nn.Linear(800, 600),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),



            nn.Linear(600, 200),
            # nn.BatchNorm1d(700),
            nn.LeakyReLU(),

            # nn.Linear(700, 400),
            # nn.BatchNorm1d(400),
            # nn.ReLU(),

            # nn.Linear(400, 200),
            # nn.ReLU(),

            nn.Linear(200, 2 * self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 200),
            nn.LeakyReLU(),


            nn.Linear(200, 600),
            nn.LeakyReLU(),

            # nn.Linear(400, 700),
            # nn.ReLU(),

            # nn.Linear(700, 1000),
            # nn.ReLU(),


            nn.Linear(600, 800),
            nn.LeakyReLU(),


            nn.Linear(800, self.input_shape),
            nn.Sigmoid

        )



    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        mu_tanh = torch.tanh(mu)
        z_min_ref = -3.3010
        z_max_ref = 3.8857
        mu = ((mu_tanh + 1) * (z_max_ref - z_min_ref) / 2) + z_min_ref
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