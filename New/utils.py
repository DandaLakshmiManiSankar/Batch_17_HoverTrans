import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2


# ===============================
# Gaussian Noise (FIXED)
# ===============================
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=10.0, p=0.5):
        self.mean = mean
        self.variance = variance
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img).astype(np.float32)
            noise = np.random.normal(self.mean, self.variance, img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)
            img = Image.fromarray(img.astype('uint8')).convert('L')
        return img


# ===============================
# Blur (Ultrasound-safe)
# ===============================
class AddBlur(object):
    def __init__(self, kernel=3, p=0.3):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            img = cv2.GaussianBlur(img, (self.kernel, self.kernel), 0)
            img = Image.fromarray(img.astype('uint8')).convert('L')
        return img


# ===============================
# Dataset
# ===============================
class Custom_Dataset(Dataset):
    def __init__(self, root, transform, csv_path):
        super().__init__()
        self.root = root
        self.transform = transform
        self.info = pd.read_csv(csv_path)

    def __getitem__(self, index):
        row = self.info.iloc[index]
        file_name = row['name']
        label = row['label']

        file_path = glob(self.root + '/' + file_name)[0]
        img = Image.open(file_path)

        if self.transform:
            img = self.transform(img)

        return {
            'imgs': img,
            'labels': torch.tensor(label, dtype=torch.long),
            'names': file_name.split('.')[0]
        }

    def __len__(self):
        return len(self.info)


# ===============================
# Dataset Loader
# ===============================
def get_dataset(imgpath, csvpath, img_size, mode='train'):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),

        # Ultrasound-safe augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),

        AddGaussianNoise(variance=8.0, p=0.5),
        AddBlur(kernel=3, p=0.3),

        transforms.ToTensor(),

        # Normalization (important!)
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform = train_transform if mode == 'train' else test_transform
    return Custom_Dataset(imgpath, transform, csvpath)


# ===============================
# Confusion Matrix
# ===============================
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += 1
    return conf_matrix
