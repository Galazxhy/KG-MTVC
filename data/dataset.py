import os
import json
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    RandomHorizontalFlip,
    RandomAffine,
    RandomRotation,
    RandomResizedCrop,
)
from PIL import Image
from config import config
from utils import utils


class HMDB_Dataset(Dataset):
    def __init__(self, root):
        super(HMDB_Dataset, self).__init__()
        spa_root = os.path.join(root, "frames")
        tem_root = os.path.join(root, "flow")

        ### label file
        label_dict = {}
        label_class = pd.read_csv(os.path.join(root, "label_class.csv"))
        with open(os.path.join(root, "labels.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                label = line.split(" ")[1].strip()
                c_idx, l_idx = (
                    label_class[label_class["label"] == label]["c_idx"],
                    label_class[label_class["label"] == label]["l_idx"],
                )
                c_idx = int(c_idx.values)
                label_ts = torch.zeros(config.MTL_classes)
                label_ts[c_idx] = torch.tensor(l_idx.values)
                label_dict[line.split(" ")[0]] = label_ts

        self.spa_paths = []
        self.tem_x_paths = []
        self.tem_y_paths = []
        self.spa_next_paths = []
        self.labels = []
        self.grade_labels = []

        dirs = os.listdir(spa_root)
        for dir in dirs:
            if dir.split("_")[-2] == "test":
                continue
            spa_files = os.listdir(os.path.join(spa_root, dir))
            spa_path, tem_x_path, tem_y_path, line = [], [], [], []
            for i, file in enumerate(spa_files):
                if (
                    int(file.split(".")[0]) > 0
                    and int(file.split(".")[0]) <= config.seq_length + 1
                ):
                    spa_path.append(os.path.join(os.path.join(spa_root, dir), file))
                    tem_x_file, tem_y_file = (
                        str(int(file.split(".")[0]) - 1) + "_x.jpg",
                        str(int(file.split(".")[0]) - 1) + "_y.jpg",
                    )
                    tem_x_path.append(
                        os.path.join(os.path.join(tem_root, dir), tem_x_file)
                    )
                    tem_y_path.append(
                        os.path.join(os.path.join(tem_root, dir), tem_y_file)
                    )
            self.spa_paths.append(spa_path)
            self.tem_x_paths.append(tem_x_path)
            self.tem_y_paths.append(tem_y_path)

            self.labels.append(label_dict[dir])

    def __len__(self):
        return len(self.spa_paths)

    def __getitem__(self, idx):
        spa_files = self.spa_paths[idx]
        tem_x_files = self.tem_x_paths[idx]
        tem_y_files = self.tem_y_paths[idx]
        label = self.labels[idx]

        trans = Compose(
            [
                ToTensor(),
                Resize((224, 224)),
            ]
        )

        spa_imgs = None
        tem_x_imgs = None
        tem_y_imgs = None
        for i in range(len(spa_files)):
            spa_imgs = utils.ts_append(spa_imgs, trans(Image.open(spa_files[i])))
            if i < len(spa_files) - 1:
                tem_x_imgs = utils.ts_append(
                    tem_x_imgs, trans(Image.open(tem_x_files[i]))
                )
                tem_y_imgs = utils.ts_append(
                    tem_y_imgs, trans(Image.open(tem_y_files[i]))
                )

        return (
            spa_imgs,
            torch.cat([tem_x_imgs, tem_y_imgs], dim=1),
            label,
            torch.tensor(0.0),
        )
