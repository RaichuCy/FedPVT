####################################################################################################
# 功能：使用GTSRB数据集
# 作者：洪建华
# 版本：创建——20250207_1855
#       编写加载数据集函数代码——20250207_1907
####################################################################################################
import torchvision.transforms as transforms
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class GTSRB(Dataset):
    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir
        self.sub_directory = 'train' if train else 'test'
        self.csv_file_name = 'train_data.csv' if train else 'test_data.csv'
        csv_file_path = os.path.join(root_dir, self.sub_directory, self.csv_file_name)
        self.csv_data = pd.read_csv(csv_file_path)
        self.transform = transform

        # 用于存储图片和标签
        self.data = []
        self.targets = []

        # 加载图片和标签
        for idx in range(len(self.csv_data)):
            img_path = os.path.join(self.root_dir, self.sub_directory, self.csv_data.iloc[idx, 0])
            img = Image.open(img_path)
            classId = self.csv_data.iloc[idx, 1]
            if self.transform:
                img = self.transform(img)

            # 将处理后的图片和标签分别存储
            self.data.append(img)  # 图片存储为 numpy 数组
            self.targets.append(classId)  # 标签存储为 Python list

        # 转换为NumPy数组
        self.data = np.array(self.data)
        if self.data.dtype == 'object':
            self.data = torch.stack([x for x in self.data])

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def loadGTSRB(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = GTSRB(root_dir=dataset_path, train=True, transform=transform)
    testset = GTSRB(root_dir=dataset_path, train=False, transform=transform)
    return trainset, testset


####################################################################################################
####################################################################################################