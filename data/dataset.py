import os
import json
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from config import config
from utils import utils


class Zinc_Dataset(Dataset):
    def __init__(self, root):
        super(Zinc_Dataset, self).__init__()
        spa_root = os.path.join(root, "frames")
        tem_root = os.path.join(root, "flow")

        ### grade label file
        l_grade_path = os.path.join(root, "time_label.txt")
        with open(l_grade_path, "r") as f:
            map = [
                [item.strip().split(",")[0] + ".mp4", float(item.strip().split(",")[1])]
                for item in f.readlines()
            ]
            key = [x[0] for x in map]
            val = [x[1] for x in map]
            map_dict = dict(zip(key, val))

        ### indirect label file
        json_path = os.path.join(root, "flotation_indirect_label.json")
        with open(json_path, "r") as file:
            json_data = json.load(file)
        annotations = {}
        for line in json_data:
            result_data = json.loads(line["result"])
            annotations_ret_list = result_data["annotations"][0]["result"]
            results = []
            for result in annotations_ret_list:
                results.append(result["value"])
            annotations[line["fileName"]] = results

        self.spa_paths = []  # 当前帧
        self.tem_x_paths = []  # 当前帧x方向光流
        self.tem_y_paths = []  # 当前帧y方向光流
        self.spa_next_paths = []  # 下一帧
        self.labels = []  # 关键特征标签
        self.grade_labels = []  # 品位标签

        dirs = os.listdir(spa_root)
        for dir in dirs:
            spa_files = os.listdir(os.path.join(spa_root, dir))  # 视频
            spa_path, tem_x_path, tem_y_path, line = [], [], [], []
            for file in spa_files:
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

            for result in annotations[dir + ".mp4"]:
                if list(result.keys())[0] == "slurryDensity":
                    if int(result["slurryDensity"][0]) == 0:
                        line.append(2)
                    if int(result["slurryDensity"][0]) == 1:
                        line.append(1)
                    if int(result["slurryDensity"][0]) == 2:
                        line.append(0)
                # if list(result.keys())[0] == "frothDepth":
                #     line.append(int(result["frothDepth"][0]))
                if list(result.keys())[0] == "frothViscosity":
                    line.append(int(result["frothViscosity"][0]))
                if list(result.keys())[0] == "pulpHeight":
                    line.append(int(result["pulpHeight"][0]))

            self.labels.append(line)
            self.grade_labels.append(map_dict[dir + ".mp4"])

    def __len__(self):
        return len(self.spa_paths)

    def __getitem__(self, idx):
        spa_files = self.spa_paths[idx]
        tem_x_files = self.tem_x_paths[idx]
        tem_y_files = self.tem_y_paths[idx]
        label = self.labels[idx]
        grade_label = self.grade_labels[idx]

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
            torch.tensor(label),
            torch.tensor(grade_label),
        )


class HMDB_Dataset(Dataset):
    def __init__(self, root):
        super(HMDB_Dataset, self).__init__()
        spa_root = os.path.join(root, "frames")
        tem_root = os.path.join(root, "flow")

        ### label file
        label_dict = {}
        label_class = pd.read_csv(os.path.join(root, "labels_class.csv"))
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

        self.spa_paths = []  # 当前帧
        self.tem_x_paths = []  # 当前帧x方向光流
        self.tem_y_paths = []  # 当前帧y方向光流
        self.spa_next_paths = []  # 下一帧
        self.labels = []  # 关键特征标签
        self.grade_labels = []  # 品位标签

        dirs = os.listdir(spa_root)
        for dir in dirs:
            if dir.split("_")[-2] == "test":
                continue
            spa_files = os.listdir(os.path.join(spa_root, dir))  # 视频
            spa_path, tem_x_path, tem_y_path, line = [], [], [], []
            for file in spa_files:
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
                Resize((256, 256)),
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

        print(spa_imgs.shape, tem_x_imgs.shape, tem_y_imgs.shape, label.shape)
        return (
            spa_imgs,
            torch.cat([tem_x_imgs, tem_y_imgs], dim=1),
            label,
            torch.tensor(0.0),
        )
