import torch
import torch.nn as nn
from einops import rearrange

from utils import utils
from config import config
import torchvision.models as models


class Reshape(nn.Module):
    """To Use Reshape in nn.Sequential/'
    ---
    """

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.contiguous().view((x.size(0),) + self.shape)


class TS_AE(nn.Module):
    def __init__(self, latent_dim):
        super(TS_AE, self).__init__()
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Reshape(32 * 32 * 32),
        )
        self.temporal_enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Reshape(32 * 32 * 32),
        )

        self.spatial_fc = nn.Linear(32 * 32 * 32, latent_dim)
        self.temporal_fc = nn.Linear(32 * 32 * 32, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32 * 32 * 32),
            nn.ReLU(),
            Reshape(32, 32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.tem2spa = utils.Cross_Attention(
            latent_dim, config.seq_length, config.seq_length, 4, 64
        )
        self.spa2tem = utils.Cross_Attention(
            latent_dim, config.seq_length, config.seq_length, 4, 64
        )

    def forward(self, x, flow):
        hid_spa = self.spatial_enc(x)
        hid_spa = self.spatial_fc(hid_spa)

        hid_tem = self.temporal_enc(flow)
        hid_tem = self.temporal_fc(hid_tem)

        hid_spa_temp = self.tem2spa(hid_tem, hid_spa)
        hid_tem_temp = self.spa2tem(hid_spa, hid_tem)

        hid = rearrange(hid_tem_temp + hid_spa_temp, "b s d -> (b s) d")

        rec_x = self.dec(hid)
        return rec_x

    def embedding(self, x, flow):
        hid_spa = self.spatial_enc(x)
        hid_spa = self.spatial_fc(hid_spa)

        hid_tem = self.temporal_enc(flow)
        hid_tem = self.temporal_fc(hid_tem)

        hid_spa_temp = self.tem2spa(hid_tem, hid_spa)
        hid_tem_temp = self.spa2tem(hid_spa, hid_tem)

        hid = rearrange(hid_tem_temp + hid_spa_temp, "b s d -> (b s) d")

        return hid
