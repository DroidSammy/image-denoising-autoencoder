import os
import cv2
import torch
from torch.utils.data import Dataset

class SIDD_Dataset(Dataset):
    def __init__(self, root, size=256):
        self.noisy_dir = os.path.join(root, "noisy")
        self.clean_dir = os.path.join(root, "clean")
        self.files = os.listdir(self.noisy_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        noisy = cv2.imread(os.path.join(self.noisy_dir, name))
        clean = cv2.imread(os.path.join(self.clean_dir, name))

        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        noisy = cv2.resize(noisy, (self.size, self.size))
        clean = cv2.resize(clean, (self.size, self.size))

        noisy = torch.from_numpy(noisy).permute(2,0,1).float() / 255.0
        clean = torch.from_numpy(clean).permute(2,0,1).float() / 255.0

        return noisy, clean
