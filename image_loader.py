from torch.utils.data import DataLoader, random_split
import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        print(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([transforms.Resize((176, 176)),
                                transforms.ToTensor()])

num_workers = 4
batch_size = 128

# dataset = CustomImageDataset(
# annotations_file="animals.csv", img_dir="animals_all", transform=transform)

dataset = datasets.ImageFolder('animals_dataset', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Create random train-test split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Optionally, you can create DataLoader objects for train and test sets to iterate over them in batches during training/testing
trainloader_animals = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader_animals = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# _________________________________________

# dataset = CustomImageDataset(
#     annotations_file="vehicles.csv", img_dir="vehicles_all", transform=transform)

dataset = datasets.ImageFolder('vehicles_dataset', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Create random train-test split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Optionally, you can create DataLoader objects for train and test sets to iterate over them in batches during training/testing
trainloader_vehicles = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader_vehicles = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
