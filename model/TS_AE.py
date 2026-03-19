import torch
import torch.nn as nn
from einops import rearrange

from utils import utils
from config import config
import torchvision.models as models


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.contiguous().view((x.size(0),) + self.shape)


class TS_AE(nn.Module):
    def __init__(self, latent_dim):
        super(TS_AE, self).__init__()
        spa_out = nn.Linear(512, latent_dim)

        self.spatial_enc = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )
        # print(self.spatial_enc)
        self.spatial_enc.fc = spa_out
        conv1 = nn.Conv2d(
            2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        tem_out = nn.Linear(512, latent_dim)
        self.temporal_enc = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.temporal_enc.conv1 = conv1
        self.temporal_enc.fc = tem_out

        self.spatial_enc.to(torch.device(config.device))
        self.temporal_enc.to(torch.device(config.device))

        # self.spatial_fc = nn.Linear(latent_dim, latent_dim)
        # self.temporal_fc = nn.Linear(latent_dim, latent_dim)
        self.spatial_mu = nn.Linear(latent_dim, latent_dim)
        self.spatial_logvar = nn.Linear(latent_dim, latent_dim)
        self.temporal_mu = nn.Linear(latent_dim, latent_dim)
        self.temporal_logvar = nn.Linear(latent_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.ReLU(),
            Reshape(256, 14, 14),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.tem2spa = utils.Cross_Attention(
            latent_dim, config.seq_length, config.seq_length, 4, 64
        )
        self.spa2tem = utils.Cross_Attention(
            latent_dim, config.seq_length, config.seq_length, 4, 64
        )

    def forward(self, x, flow):
        hid_spa = self.spatial_enc(x)
        # hid_spa = self.spatial_fc(hid_spa)
        spa_mu = self.spatial_mu(hid_spa)
        spa_logvar = self.spatial_logvar(hid_spa)

        hid_tem = self.temporal_enc(flow)
        # hid_tem = self.temporal_fc(hid_tem)
        tem_mu = self.temporal_mu(hid_tem)
        tem_logvar = self.temporal_logvar(hid_tem)

        spa_z = self.reparameterize(spa_mu, spa_logvar)
        tem_z = self.reparameterize(tem_mu, tem_logvar)

        hid_spa_temp = self.tem2spa(tem_z, spa_z)
        hid_tem_temp = self.spa2tem(spa_z, tem_z)

        hid = rearrange(hid_tem_temp + hid_spa_temp, "b s d -> (b s) d")

        rec_x = self.dec(hid)
        return rec_x, spa_mu, spa_logvar, tem_mu, tem_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def embedding(self, x, flow):
        hid_spa = self.spatial_enc(x)
        # hid_spa = self.spatial_fc(hid_spa)
        spa_mu = self.spatial_mu(hid_spa)
        spa_logvar = self.spatial_logvar(hid_spa)
        spa_z = self.reparameterize(spa_mu, spa_logvar)

        hid_tem = self.temporal_enc(flow)
        # hid_tem = self.temporal_fc(hid_tem)
        tem_mu = self.temporal_mu(hid_tem)
        tem_logvar = self.temporal_logvar(hid_tem)
        tem_z = self.reparameterize(tem_mu, tem_logvar)

        hid_spa_temp = self.tem2spa(tem_z, spa_z)
        hid_tem_temp = self.spa2tem(spa_z, tem_z)

        hid = rearrange(hid_tem_temp + hid_spa_temp, "b s d -> (b s) d")

        return hid, spa_mu, spa_logvar, tem_mu, tem_logvar

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
