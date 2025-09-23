import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import exp
from config import config
from torch.utils.data import random_split, DataLoader
from data.dataset import Zinc_Dataset, HMDB_Dataset


def log_results(result_dict):
    path = result_dict["Path"]
    with open(path + f"/result_{result_dict['mode']}.txt", "a") as f:
        for key, value in result_dict.items():
            if key == "best_net":
                torch.save(value, path + f"/best_net_{result_dict['mode']}.pth")
            elif key != "best_net":
                f.write(key + ": " + str(value) + "\n")


def get_data():
    if config.data_name == "Zinc":
        dataset = Zinc_Dataset(config.data_root)
    elif config.data_name == "HMDB":
        dataset = HMDB_Dataset(config.data_root)

    trn_set, val_set, tst_set = random_split(
        dataset,
        [
            round(0.6 * len(dataset)),
            round(0.2 * len(dataset)),
            len(dataset) - round(0.6 * len(dataset)) - round(0.2 * len(dataset)),
        ],
    )
    trn_loader = DataLoader(
        trn_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    tst_loader = DataLoader(
        tst_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    return trn_loader, val_loader, tst_loader


"""
description: List like 'Append' tool for tensor datatype
param {append a with b} a
param {append a with b} b
return {Appending result}
"""


def ts_append(a, b):
    if a is None:
        return b.unsqueeze(0)
    else:
        return torch.cat([a, b.unsqueeze(0)], dim=0)


class EarlyStopping:
    """
    description: Early stopping
    method {Initializing} __init__()
    method {Iteration} __call__()
    method {Reset} reset()
    """

    """
    description: Initializing
    param {Class EarlyStopping} self
    param {How long to wait after last time validation loss improved.} patience
    param {Minimum change in the monitored quantity to qualify as an improvement.} delta
    """

    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.delta = delta

    """
    description: Iteration step
    param {Class EarlyStopping} self
    param {Results of training} results
    """

    def __call__(self, val_loss, results):
        score = -val_loss
        if self.bestScore is None:
            self.bestScore = score
        elif score < self.bestScore + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.bestScore = score
            self.counter = 0

    """
    description: Reset parameters
    param {Class EarlyStopping} self
    """

    def reset(self):
        self.bestScore = None
        self.earlyStop = False
        self.counter = 0


class Cross_Attention(nn.Module):
    def __init__(self, dim, seq_length_q, seq_length_k, heads=4, dim_head=64):
        """
        dim: input embeddings
        heads: number of attention head
        dim_head: number of attention head dimension
        """
        super(Cross_Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.seq_length_q = seq_length_q
        self.seq_length_k = seq_length_k

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.5))

    def forward(self, x, context):
        """
        x: Q
        context: K/V
        """
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "(b s) (h d) -> (b h) s d", h=h, s=self.seq_length_q)
        k = rearrange(k, "(b s) (h d) -> (b h) s d", h=h, s=self.seq_length_k)
        v = rearrange(v, "(b s) (h d) -> (b h) s d", h=h, s=self.seq_length_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)

        out = rearrange(out, "(b h) s d -> b s (h d)", h=h, s=self.seq_length_q)

        return self.to_out(out)


class MultiLossLayer(nn.Module):
    """
    计算自适应损失权重
    implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(self, num_loss):
        """
        Args:
            num_loss (int): number of multi-task loss
        """
        super(MultiLossLayer, self).__init__()
        # sigmas^2 (num_loss,)
        # uniform init
        # 从均匀分布U(a, b)中生成值，填充输入的张量或变量，其中a为均匀分布中的下界，b为均匀分布中的上界
        self.sigmas_sq = nn.Parameter(
            nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0), requires_grad=True
        )

    def get_loss(self, loss_set):
        """
        Args:
            loss_set (Tensor): multi-task loss (num_loss,)
        """
        # 1/2σ^2
        # (num_loss,)
        # self.sigmas_sq -> tensor([0.9004, 0.4505]) -> tensor([0.6517, 0.8004]) -> tensor([0.7673, 0.6247])
        # 出现左右两个数随着迭代次数的增加，相对大小交替变换
        factor = torch.div(1.0, torch.mul(2.0, self.sigmas_sq))
        # loss part (num_loss,)
        loss_part = torch.sum(torch.mul(factor, loss_set))
        # regular part 正则项，防止某个σ过大而引起训练严重失衡。
        regular_part = torch.sum(torch.log(self.sigmas_sq))

        loss = loss_part + regular_part

        return loss


# 计算一维高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


# 创建高斯核
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# SSIM类
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = (
                create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel
        return ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
        )
