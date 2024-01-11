# get dir sort file name, get first 12000 images


import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from train import VGG16_LargeFOV
from torch.utils.data import DataLoader



colorDict = {
    "alveolous": [0, 255, 255],
    "luad1": [0, 255, 0],
    "luad2": [0, 0, 255],
    "luad3": [255, 255, 0],
    "luad4": [255, 0, 0],
    "background": [255, 255, 255],
}


def RGBtoOneHot(im, colorDict):
    arr = torch.zeros(im.shape[-2:])  ## rgb shape: (3,h,w); arr shape: (h,w)
    for label, color in enumerate(colorDict.values()):
        color = torch.tensor(color)
        if label < len(colorDict.values()):
            arr[torch.all(im == color, dim=0)] = label
    arr = arr.unsqueeze(0)
    return arr


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

    def __getitem__(self, idx):
        print(self.root, self.img_dir, self.im_list[idx])
        img_path = os.path.join(self.root, self.img_dir, self.im_list[idx])
        image = read_image(img_path)
        label_path = os.path.join(self.root, self.label_dir, self.im_list[idx])
        label = read_image(label_path)
        label = RGBtoOneHot(label, colorDict)
        
        # label = label.permute(1, 2, 0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



root = r"/mnt/d/GlassAI_data"

train_ratio = 0.9
size = 500
train_size = int(size*train_ratio)
train_range , val_range = (0,train_size), (train_size,size)
train = LungImageDataset(root,  size=train_range)
val = LungImageDataset(root, size=val_range)

vgg=VGG16_LargeFOV(num_classes=len(colorDict.keys()))

train = DataLoader(train, batch_size=50, shuffle=True)
val = DataLoader(val, batch_size=50, shuffle=True)

vgg.train(train_data=train, test_data=val)

# d = LungImageDataset(root, size=(0, 500))


# colors = {}
# # Convert the list to a NumPy array for faster processing

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(d, batch_size=50, shuffle=False)

# for im, label in train_dataloader:
#     label = label.reshape(-1, 3)
#     # Get unique RGB values and their counts
#     uniques, counts = torch.unique(label, dim=0, return_counts=True)
#     for unique, count in zip(uniques, counts):
#         unique = str(unique)
#         colors[unique] = colors.get(unique, 0) + count
#     print(label.shape)

# print(colors)


# load images as dataset
# train
