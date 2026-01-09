import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from model.MC_FRGE import MC_FRGE
import pandas as pd
from einops import rearrange

from pytorchvideo.models import resnet


class KG_MTVC(nn.Module):
    def __init__(self, tsae_model, root):
        super(KG_MTVC, self).__init__()
        self.tsae_model = tsae_model

        self.MC_FRGE = MC_FRGE(root)

    def forward(self, imgs, flows):
        assert self.tsae_model != None
        emb, _, _, _, _ = self.tsae_model.embedding(imgs, flows)
        out = self.MC_FRGE(emb)
        return out


class CrossStitch(nn.Module):
    def __init__(self, task_num):
        super(CrossStitch, self).__init__()
        self.alpha = nn.Parameter(torch.randn(task_num, task_num), requires_grad=True)

    def forward(self, X):
        out = torch.einsum("ij,jbd->ibd", self.alpha, X)
        return out  # out: [task_num, batch_size, feature_dim]


class Resnet3D(nn.Module):
    def __init__(self, root):
        super(Resnet3D, self).__init__()
        self.root = root

        self.resnet = resnet.create_resnet(
            input_channel=3,
            model_depth=50,
            model_num_class=config.latent_dim,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

        self.multi_class_list = self.multi_class_construct()
        self.classifier_modules = nn.ModuleList()
        for key in self.multi_class_list.keys():
            self.classifier_modules.append(
                nn.ModuleList(
                    [
                        nn.Linear(config.latent_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(128, len(self.multi_class_list[key])),
                    ]
                )
            )

    def forward(self, x):
        x = rearrange(x, "(b t) c w h -> b c t w h", t=config.seq_length)
        hid = self.resnet(x)

        class_out = []
        for i in range(len(self.classifier_modules)):
            temp_hid = hid.clone()
            for j in range(len(self.classifier_modules[i])):
                temp_hid = self.classifier_modules[i][j](temp_hid)
            class_out.append(F.log_softmax(temp_hid, dim=1))

        return class_out

    def multi_class_construct(self):
        label_class = pd.read_csv(os.path.join(self.root, "label_class.csv")).values
        multi_class_list = {}
        for line in label_class:
            if line[0] not in multi_class_list.keys():
                multi_class_list[line[0]] = [line[2]]
            else:
                multi_class_list[line[0]].append(line[2])
        return multi_class_list


class X3D(nn.Module):
    def __init__(self, root):
        super(X3D, self).__init__()
        self.root = root

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", model="x3d_s", pretrained=True
        )
        self.model.blocks[5].proj = nn.Linear(2048, config.latent_dim)

        self.multi_class_list = self.multi_class_construct()
        self.classifier_modules = nn.ModuleList()
        for key in self.multi_class_list.keys():
            self.classifier_modules.append(
                nn.ModuleList(
                    [
                        nn.Linear(config.latent_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(128, len(self.multi_class_list[key])),
                    ]
                )
            )

    def forward(self, x):
        x = rearrange(x, "(b t) c w h -> b c t w h", t=config.seq_length)
        hid = self.model(x)

        class_out = []
        for i in range(len(self.classifier_modules)):
            temp_hid = hid.clone()
            for j in range(len(self.classifier_modules[i])):
                temp_hid = self.classifier_modules[i][j](temp_hid)
            class_out.append(F.log_softmax(temp_hid, dim=1))

        return class_out

    def multi_class_construct(self):
        label_class = pd.read_csv(os.path.join(self.root, "label_class.csv")).values
        multi_class_list = {}
        for line in label_class:
            if line[0] not in multi_class_list.keys():
                multi_class_list[line[0]] = [line[2]]
            else:
                multi_class_list[line[0]].append(line[2])
        return multi_class_list


class ResNet3D_CrossStitch(nn.Module):
    def __init__(self, root):
        super(ResNet3D_CrossStitch, self).__init__()
        self.root = root

        self.resnet = resnet.create_resnet(
            input_channel=3,
            model_depth=50,
            model_num_class=config.latent_dim,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

        # Cross-Stitch Classifier
        self.multi_class_list = self.multi_class_construct()
        self.fc_mtl1 = nn.ModuleList()
        self.fc_mtl2 = nn.ModuleList()
        self.fc_mtl3 = nn.ModuleList()
        for key in self.multi_class_list.keys():
            self.fc_mtl1.append(
                nn.ModuleList([nn.Linear(config.latent_dim, 128), nn.ReLU()])
            )
            self.fc_mtl2.append(
                nn.ModuleList([nn.Linear(128, 128), nn.ReLU(), nn.Dropout()])
            )
            self.fc_mtl3.append(
                nn.Linear(
                    128,
                    len(self.multi_class_list[key]),
                )
            )
        self.cross_stitch1 = CrossStitch(len(self.multi_class_list.keys()))
        self.cross_stitch2 = CrossStitch(len(self.multi_class_list.keys()))

    def forward(self, x):
        x = rearrange(x, "(b t) c w h -> b c t w h", t=config.seq_length)
        hid = self.resnet(x)

        hiddens = []
        for i in range(len(self.multi_class_list)):
            temp_hid = hid.clone()
            for j in range(len(self.fc_mtl1[i])):
                temp_hid = self.fc_mtl1[i][j](temp_hid)
            hiddens.append(temp_hid)
        hiddens = self.cross_stitch1(torch.stack(hiddens, dim=0))
        for i in range(len(self.multi_class_list)):
            temp_hid = hiddens[i].clone()
            for j in range(len(self.fc_mtl2[i])):
                temp_hid = self.fc_mtl2[i][j](temp_hid)
            hiddens[i] = temp_hid
        hiddens = self.cross_stitch2(hiddens)
        class_out = []
        for i in range(len(self.multi_class_list)):
            temp_hid = hiddens[i].clone()
            temp_hid = self.fc_mtl3[i](temp_hid)
            class_out.append(F.log_softmax(temp_hid, dim=1))

        return class_out

    def multi_class_construct(self):
        label_class = pd.read_csv(os.path.join(self.root, "label_class.csv")).values
        multi_class_list = {}
        for line in label_class:
            if line[0] not in multi_class_list.keys():
                multi_class_list[line[0]] = [line[2]]
            else:
                multi_class_list[line[0]].append(line[2])
        return multi_class_list


class X3D_CrossStitch(nn.Module):
    def __init__(self, root):
        super(X3D_CrossStitch, self).__init__()
        self.root = root

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", model="x3d_s", pretrained=True
        )
        self.model.blocks[5].proj = nn.Linear(2048, config.latent_dim)

        # Cross-Stitch Classifier
        self.multi_class_list = self.multi_class_construct()
        self.fc_mtl1 = nn.ModuleList()
        self.fc_mtl2 = nn.ModuleList()
        self.fc_mtl3 = nn.ModuleList()
        for key in self.multi_class_list.keys():
            self.fc_mtl1.append(
                nn.ModuleList([nn.Linear(config.latent_dim, 128), nn.ReLU()])
            )
            self.fc_mtl2.append(
                nn.ModuleList([nn.Linear(128, 128), nn.ReLU(), nn.Dropout()])
            )
            self.fc_mtl3.append(
                nn.Linear(
                    128,
                    len(self.multi_class_list[key]),
                )
            )
        self.cross_stitch1 = CrossStitch(len(self.multi_class_list.keys()))
        self.cross_stitch2 = CrossStitch(len(self.multi_class_list.keys()))

    def forward(self, x):
        x = rearrange(x, "(b t) c w h -> b c t w h", t=config.seq_length)
        hid = self.model(x)

        hiddens = []
        for i in range(len(self.multi_class_list)):
            temp_hid = hid.clone()
            for j in range(len(self.fc_mtl1[i])):
                temp_hid = self.fc_mtl1[i][j](temp_hid)
            hiddens.append(temp_hid)
        hiddens = self.cross_stitch1(torch.stack(hiddens, dim=0))
        for i in range(len(self.multi_class_list)):
            temp_hid = hiddens[i].clone()
            for j in range(len(self.fc_mtl2[i])):
                temp_hid = self.fc_mtl2[i][j](temp_hid)
            hiddens[i] = temp_hid
        hiddens = self.cross_stitch2(hiddens)
        class_out = []
        for i in range(len(self.multi_class_list)):
            temp_hid = hiddens[i].clone()
            temp_hid = self.fc_mtl3[i](temp_hid)
            class_out.append(F.log_softmax(temp_hid, dim=1))

        return class_out

    def multi_class_construct(self):
        label_class = pd.read_csv(os.path.join(self.root, "label_class.csv")).values
        multi_class_list = {}
        for line in label_class:
            if line[0] not in multi_class_list.keys():
                multi_class_list[line[0]] = [line[2]]
            else:
                multi_class_list[line[0]].append(line[2])
        return multi_class_list
