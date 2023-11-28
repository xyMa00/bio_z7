# encoding: utf-8
import os
import math
import numpy as np
from torchvision.datasets import VisionDataset

# ds_dir = "/home/datacenter/ds/Forebrain"
ds_dir='Forebrain'

class ForeBrain(VisionDataset):
    def __init__(self):
        # file_path = os.path.join(ds_dir, "data_scale.txt")
        file_path = os.path.join(ds_dir, "data_ED.txt")
        label_path = os.path.join(ds_dir, "labels.txt")
        self.label = list()
        data = list()
        with open(file_path) as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    continue
                arr = line.split("\t")
                data.append([int(t) for t in arr[1:]])
        self.data = np.asarray(data).T
        print(self.data.shape)
        label_origin = list()
        label_dict = dict()
        label_index = 0
        with open(label_path) as f:
            for index, line in enumerate(f.readlines()):
                label = line.split("\t")[1]
                label_origin.append(label)
                if label not in label_dict:
                    label_dict[label] = label_index
                    label_index += 1
        self.cluster_num = len(label_dict)
        self.labels = [label_dict[a] for a in label_origin]
        row = self.data.shape[0]
        col = self.data.shape[1]
        self.col_sqrt = math.floor(math.sqrt(col))
        # jaccard = np.zeros((row, row))
        # for i in range(row):
        #     print(i)
        #     for j in range(i + 1, row):
        #         v = sum(self.data[i] == self.data[j]) / col
        #         jaccard[i][j] = v
        #         jaccard[j][i] = v
        # self.jaccard = jaccard

    def __getitem__(self, index):
        n = self.col_sqrt * self.col_sqrt
        a = self.data[index][: n].reshape(self.col_sqrt, self.col_sqrt)
        a = a[np.newaxis, :, :]
        return a.astype(np.float32), 0

    def __len__(self):
        return len(self.data)
