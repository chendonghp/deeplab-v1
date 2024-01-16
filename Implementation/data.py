# get dir sort file name, get first 12000 images

from typing import Tuple
import os
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import numpy as np
from numpy._typing import NDArray

colorDict = OrderedDict(
    {
        "background": [255, 255, 255],
        "luad1": [0, 255, 0],
        "luad2": [0, 0, 255],
        "luad3": [255, 255, 0],
        "luad4": [255, 0, 0],
        "alveolous": [0, 255, 255],
    }
)


# https://medium.com/@noah.vandal/creating-a-label-class-matrix-from-an-rgb-mask-for-segmentation-training-in-python-2ddceba459cb
def rgb2onehot(im: NDArray, colorDict: dict) -> NDArray:
    arr = np.zeros(im.shape[:2])  ## rgb shape: (3,h,w); arr shape: (h,w)
    for label, color in enumerate(colorDict.values()):
        color = np.asarray(color)
        arr[np.all(im == color, axis=-1)] = label
    # print(arr.dtype)
    return arr


def onehot2rgb(im: NDArray, colorDict: dict) -> NDArray:
    arr = np.empty((*im.shape, 3))
    for label, color in enumerate(colorDict.values()):
        arr[im == np.asarray(label)] = np.asarray(color)
    return arr.astype(np.uint8)


class LungImageDataset(Dataset):
    def __init__(
        self,
        root,
        img_dir="Images",
        label_dir="Labels",
        transform=None,
        target_transform=None,
        size=(0, 12000),
    ):
        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.im_list = []
        for f in os.listdir(os.path.join(root, img_dir)):
            if f.split(".")[-1] == "png":
                self.im_list.append(f)
        if size:
            self.im_list = self.im_list[size[0] : size[1]]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx) -> Tuple[Image.Image, Image.Image]:
        # print(self.root, self.img_dir, self.im_list[idx])
        img_path = os.path.join(self.root, self.img_dir, self.im_list[idx])
        image = Image.open(img_path)
        label_path = os.path.join(self.root, self.label_dir, self.im_list[idx])
        label = Image.open(label_path)
        label = np.asarray(label)
        label = Image.fromarray(rgb2onehot(label, colorDict))
        if self.transform:
            image, label = self.transform(image, label)
        return image, label


print("test")
