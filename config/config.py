# Train
# pretrained = "./log/exp_1/rep_0/best_net_TS-AE.pth"
pretrained = None
MTL_classes = 5
rep = 10
epoch = 200
batch_size = 12
device = "cuda:1"

# Optimizer
lr = 0.0001
wd = 1e-5

# data
data_name = "Zinc"
data_root = "../Data/Zinc"
seq_length = 30

# Early Stop
patience = 20
delta = 1e-4

# TS-AE
latent_dim = 256
pretrn_epoch = 30
pretrn_lr = 0.002
pretrn_wd = 1e-5
