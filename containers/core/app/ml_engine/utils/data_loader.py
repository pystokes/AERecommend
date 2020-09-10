#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from pathlib import Path
import random
import re

import cv2
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

class CreateDataLoader(object):

    @classmethod
    def build_for_train(self, config):

        img_list = list(Path(config.train.input_dir).glob('*.jpg'))
        random.shuffle(img_list)
        split_idx = int(len(img_list) * config.train.train_split_ratio)
        train_img_list = img_list[:split_idx]
        valid_img_list = img_list[split_idx:]

        # Dataset
        train_dataset = BatchDataset(train_img_list,
                                     config,
                                     transform=transforms.Compose(
                                         [
                                             transforms.ToTensor(),
                                         ]
                                     ))

        valid_dataset = BatchDataset(valid_img_list,
                                     config,
                                     transform=transforms.Compose(
                                         [
                                             transforms.ToTensor(),
                                         ]
                                     ))

        # Data loader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=config.train.shuffle)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=False)

        return train_loader, valid_loader


    @classmethod
    def build_for_detect(self, config, x_dir):

        inputs = [img_path for img_path in Path(x_dir).glob('*') if re.fullmatch('.jpg|.jpeg|.png', img_path.suffix.lower())]

        dataset = BatchDataset(inputs=inputs,
                               config=config,
                               transform=transforms.Compose(
                                   [
                                       transforms.ToTensor(),
                                   ]
                               ))

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0)

        return data_loader


class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, config, transform=None):

        self.inputs = inputs
        self.resize = config.model.input_size
        self.transform = transform

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        x = cv2.imread(str(self.inputs[idx]))
        x = cv2.resize(x, (self.resize, self.resize)).astype(np.float32)
        x = x[:, :, ::-1].copy() # Reorder from BGR to RGB

        if self.transform:
            x = self.transform(x)

        return str(self.inputs[idx]), x


if __name__ == '__main__':
    pass
