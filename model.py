import torch.nn as nn
import torch
from torch import Tensor
from torch._inductor.ir import View
from torch.autograd import Variable
from torch.nn import init


class VariationalAutoencodermodel(nn.Module):
    def __init__(self, latent_dim=30):
        super(VariationalAutoencodermodel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 200, kernel_size=3),
            nn.ReLU(),
            GroupNorm(200),
            nn.Conv2d(200, 150, kernel_size=3,stride=2),
            nn.ReLU(),
            GroupNorm(150,num_groups=30),
            nn.Conv2d(150,100,kernel_size=3),
            nn.ReLU(),
            GroupNorm(100,num_groups=10),
            nn.Conv2d(100,70,kernel_size=2),
            nn.ReLU(),
            GroupNorm(70,num_groups=10),
            nn.Conv2d(70,60,kernel_size=2),
            nn.ReLU(),
            GroupNorm(60,num_groups=20),
            nn.Conv2d(60, 50, kernel_size=1),
            nn.ReLU(),
            View((-1, 50 * 1 * 1)),
            nn.Linear(50, latent_dim * 2),

        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            View((-1, 50, 1, 1)),
            nn.ConvTranspose2d(50, 100, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 150, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(150, 256, kernel_size=3),
            nn.Tanh()

        )

        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        # reconstruct the data based on the learned data representation
        y = self.decoder(z)
        # # reconstruct the images based on the learned data representation
        img = self.img_decoder(y)

        return z, y, img, mu, logvar


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

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)  # Updated to kaiming_normal_
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)



class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups ,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # normalize
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.gamma + self.beta
