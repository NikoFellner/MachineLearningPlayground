import os
import skimage.color
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.directory = "images"
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=train_mean, std=train_std)
                                                 ])
        self._transform_train = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                    tv.transforms.ToTensor(),
                                                    tv.transforms.Normalize(mean=train_mean, std=train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.directory, self.data.loc[index, 'filename'])
        img = imread(img_path)
        label = torch.tensor((int(self.data.loc[index, 'crack']), int(self.data.loc[index, 'inactive'])), dtype=torch.float32)
        img = skimage.color.gray2rgb(img)
        if self.mode == "val":
            img = self._transform(img)
        if self.mode == "train":
            img = self._transform_train(img)

        return (img, label)


