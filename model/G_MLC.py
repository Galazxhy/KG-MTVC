import os
import torch
import pandas as pd
import torch_geometric.nn as nn
import torch.nn.functional as F

from utils import utils
from config import config
from einops import rearrange


class G_MLC(torch.nn.Module):
    def __init__(self, root):
        super(G_MLC, self).__init__()
        self.root = root
        self.word_idx = pd.read_csv(os.path.join(self.root, "word_idx.csv"))

        rule_tuple = pd.read_csv(os.path.join(self.root, "rule_tuple.csv"))
        self.basic = rule_tuple["basic"].values
        self.crucial = rule_tuple["crucial"].values
        self.label = rule_tuple["label"].values

        self.multi_class_list = self.multi_class_construct()

        self.to_b = nn.Linear(
            len(self.word_idx["word"]), 256
        )  # basis concept embedding dimension: 128
        self.to_k = nn.Linear(
            len(self.word_idx["word"]), 256
        )  # crucial concept embedding dimension: 128

        self.vis2rule = utils.Cross_Attention(
            256, config.seq_length, len(rule_tuple), 4, 256
        )
        self.rule2vis = utils.Cross_Attention(
            256, len(rule_tuple), config.seq_length, 4, 256
        )

        self.G_modules = torch.nn.ModuleList()
        for key in self.multi_class_list.keys():
            self.G_modules.append(
                torch.nn.ModuleList(
                    [
                        nn.GATConv(256, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(),
                        nn.GATConv(128, 64),
                        torch.nn.Linear(64, len(self.multi_class_list[key])),
                    ]
                )
            )

    def multi_class_construct(self):
        label_class = pd.read_csv(os.path.join(self.root, "label_class.csv")).values
        multi_class_list = {}
        for line in label_class:
            if line[0] not in multi_class_list.keys():
                multi_class_list[line[0]] = [line[2]]
            else:
                multi_class_list[line[0]].append(line[2])
        return multi_class_list

    def concept2multihot(self, concepts):
        multihots = None
        for concept in concepts:
            comps = concept.split("-")
            multihot = torch.zeros(len(self.word_idx["word"]))
            for comp in comps:
                idx = [
                    int(item)
                    for item in self.word_idx[self.word_idx["word"] == comp][
                        "index"
                    ].values
                ]
                multihot[idx] = 1
            multihots = utils.ts_append(multihots, multihot)
        return multihots

    def label2class(self, labels):
        label_csv = pd.read_csv(os.path.join(self.root, "label_class.csv"))
        classes = torch.zeros(len(labels))
        for i, label in enumerate(labels):
            l, c = label.split("-")
            classes[i] = int(
                label_csv[label_csv["label_class"] == c]["c_idx"].values.mean()
            )

        return classes

    def graph_construct(self):
        basic_multihot = self.concept2multihot(self.basic)
        crucial_multihot = self.concept2multihot(self.crucial)

        label_class = self.label2class(self.label)

        adj_basic = [
            sum(torch.logical_and(a, b)) != 0
            for a in basic_multihot
            for b in basic_multihot
        ]

        adj_crucial = [
            sum(torch.logical_and(a, b)) != 0
            for a in crucial_multihot
            for b in crucial_multihot
        ]

        adj = rearrange(
            torch.logical_and(torch.stack(adj_basic), torch.stack(adj_crucial)),
            "(w h) ->w h",
            h=len(basic_multihot),
        )

        mask = None
        for i in range(len(self.multi_class_list.keys())):
            mask = utils.ts_append(mask, label_class == i).to(torch.int32)

        return (
            basic_multihot.to(torch.device(config.device)),
            crucial_multihot.to(torch.device(config.device)),
            adj.to(torch.device(config.device)),
            mask.to(torch.device(config.device)),
        )

    def forward(self, vis_emb):
        basic_multihot, crucial_multihot, adj, mask = self.graph_construct()
        basic_emb, crucial_emb = self.to_b(basic_multihot), self.to_k(crucial_multihot)
        rule_emb = basic_emb + crucial_emb
        # print(vis_emb.shape)
        emb = self.rule2vis(
            rule_emb.repeat(int(vis_emb.shape[0] / config.seq_length), 1), vis_emb
        )

        class_out = []
        for i in range(len(self.G_modules)):
            out_emb = None
            for data_point in emb:
                emb_temp = data_point * mask[i].unsqueeze(1).repeat(1, 256)
                for j in range(len(self.G_modules[i])):
                    if isinstance(self.G_modules[i][j], nn.GATConv):
                        emb_temp = self.G_modules[i][j](emb_temp, adj.to_sparse())
                    else:
                        emb_temp = self.G_modules[i][j](emb_temp)
                out_emb = utils.ts_append(out_emb, emb_temp)
            class_out.append(F.log_softmax(out_emb.sum(dim=1, keepdim=False), dim=1))

        return class_out
