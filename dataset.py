import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class GliomaDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]

        image_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename.replace(".png", "_mask.png"))

        image = Image.open(image_path).convert("L").resize((256, 256), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        mask = (mask > 0.5).float()

        return image, mask