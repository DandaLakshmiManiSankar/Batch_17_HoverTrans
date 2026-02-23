import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, img_size=256, ssl=False):
        self.root_dir = root_dir
        self.ssl = ssl

        self.images = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)

        if self.ssl:
            return img
        else:
            label = 0 if "benign" in self.images[idx].lower() else 1
            return img, label
