# compute image mean and std
from data import compute_mean_std, LungImageDataset
import os
import augmentation
import json

data_root = r"/mnt/d/data/"
size = 120
from torch.utils.data import DataLoader


dataset = LungImageDataset(
    data_root,
    transform=augmentation.Mask_Aug(
        [augmentation.PILImage2Tensor(), augmentation.PILMask2Tensor()]
    ),
    size=(0, size),
)
dataset_loader = DataLoader(
    dataset,
    batch_size=200,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

train_features, train_labels = next(iter(dataset_loader))
print(train_features.shape, train_labels.shape)
