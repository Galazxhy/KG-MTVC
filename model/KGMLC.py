import torch.nn as nn

from model.G_MLC import G_MLC


class KGMLC(nn.Module):
    def __init__(self, tsae_model, root):
        super(KGMLC, self).__init__()
        self.tsae_model = tsae_model
        self.g_mlc = G_MLC(root)

    def forward(self, imgs, flows):
        assert self.tsae_model != None
        emb = self.tsae_model.embedding(imgs, flows)
        out = self.g_mlc(emb)
        return out
