import torch.nn as nn
import torch
from torch import Tensor
from torch._inductor.ir import View
from torch.autograd import Variable


class VAE_GE(nn.Module):
    def __init__(self, latent_dim=50):
        super(VAE_GE, self).__init__()
        self.latent_dim=latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Conv1d(128,100,kernel_size=3, stride=2),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Conv1d(100, 50,kernel_size=3, stride=2),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Conv1d(50,25,kernel_size=3, stride=2),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Conv1d(25, 1, kernel_size=3, stride=2),
            # nn.BatchNorm1d(1),
            # nn.ReLU(),
            # nn.MaxPool1d(3, stride=3),
            nn.Flatten(),


        )

        self.fc_out = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            View((-1, 1, 50)),
            nn.ConvTranspose1d(1, 25, kernel_size=4, stride=4, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(25, 50, kernel_size=4, stride=4, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(50, 100, kernel_size=4, stride=4, padding=3, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(100, 128, kernel_size=3, stride=3, padding=2, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3, padding=2, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, stride=2, padding=4, output_padding=1),
            # nn.ConvTranspose1d(1, 1, kernel_size=1, stride=1),
            # nn.Upsample(size=58604, mode='linear', align_corners=True),
            nn.ReLU()
        )



    def forward(self, x):
        x = self.encoder(x)
        if self.fc_out is None:
            self.fc_out = nn.Linear(x.shape[1], 2 * self.latent_dim).to(x.device)
        distributions = self.fc_out(x)
        # Split the distributions into mu and logvar
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        y = self.decoder(x)

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