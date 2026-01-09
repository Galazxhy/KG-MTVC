# General
pretrained = None
# load pretrained model: pretrained = "./log/exp_1/rep_0/best_net_TS-AE.pth" #
MTL_classes = 5  # Fixed to 5
rep = 1
epoch = 500
batch_size = 16
device = "cuda:0"
model = (
    "KG_MTVC"  # Implemented: ['KG_MTVC', 'Resnet3D', 'Resnet3D_CS', 'X3D', 'X3D_CS']
)

# Optimizer
lr = 0.0001
wd = 1e-4

# Data
data_name = "HMDB"
data_root = "./Data/HMDB"
seq_length = 19

# Early Stopping
patience = 50
delta = 0

# TS-AE
latent_dim = 512
pretrn_epoch = 5
pretrn_lr = 1e-4
pretrn_wd = 1e-4

vae_beta = 0.01
