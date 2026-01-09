
## :globe_with_meridians: Knowledge-Guided Multi-Task Video Classification for Industrial Process via Fuzzy Rule Graph Embedding
This is the official implementation of `Knowledge-Guided Multi-Task Video Classification for Industrial Process via Fuzzy Rule Graph Embedding`.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0%2B-orange)]()

### :earth_asia: Environment Dependencies
```requirements
python==3.8+
torch==1.12.0+
torchvision==0.13.0+
pytorchvideo==0.1.5+
einops==0.6.0+
pandas==1.4.0+
matplotlib==3.5.0+
scikit-learn==1.0.2+
```

### :wrench: Installation
1. Clone the repository:
```bash
git clone https://github.com/Galazxhy/KG-MTVC.git
```
2. Create and activate virtual environment:
```bash
conda create -n kg-mtvc python=3.8
conda activate kg-mtvc
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### :cd: Data Preparation
1. Download the processed dataset from [Zenodo](https://zenodo.org/records/18524356)
2. Extract HMDB51 dataset to `Data/HMDB` directory.
3. Ensure directory structure contains the following subdirectories and files
    - `frames` (video frames) 
    - `flow` (optical flow) subdirectories
    - `labels.txt`
    - `label_class.csv`
    - `laebls_test.txt`
    - `rule_tuple.csv`
    - `word_idx.csv`


### :rocket: Usage
#### Training and Validating the Model
```bash
python run.py
```
#### Configuring Training Parameters
Edit `config/config.py` to customize training parameters:
```python
# General
pretrained = None
# load pretrained model: pretrained = "./log/exp_1/rep_0/best_net_TS-AE.pth" #
MTL_classes = 5  # Fixed to 5
rep = 1
epoch = 500
batch_size = 16
device = "cuda:0"
model = "KG_MTVC"  # Implemented: ['KG_MTVC', 'Resnet3D', 'Resnet3D_CS','X3D', 'X3D_CS']

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
```

#### Evaluating the Model
After training, models and results will be automatically saved in the `log/` directory.

### :file_folder: Project Structure
```
KGMLC/
├── config/                # Configuration files
│   ├── __init__.py
│   └── config.py          # Project parameter configuration
├── data/                  # Data processing module
│   ├── __init__.py
│   └── dataset.py         # Dataset loading and preprocessing
├── model/                 # Model definitions
│   ├──__init__.py
│   ├── KG_MTVC.py         # Core knowledge graph multi-task 
│   ├── Model.py           # Various backbone network 
│   └── TS_AE.py           # Temporal-Spatial Autoencoder
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── utils.py           # Helper functions and utilities
├── README.md              # Project documentation
├── LICENSE                # License
├── requirements.txt       # Dependency list
└── run.py                 # Main execution script
```

### :book: Citation
---

---
### :scroll: LICENSE
This project is licensed uder the [MIT License](./LICENSE)
* last updated: `2026/02/08`
