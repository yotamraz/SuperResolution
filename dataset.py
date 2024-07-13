import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from image_utils import image_to_tensor


class SRImageDataset(Dataset):
    def __init__(self, path_to_folder: str, img_h: int, img_w: int, scaling_factor: int = 4):

        if len(os.listdir(path_to_folder)) == 0:
            raise FileNotFoundError(f"No images found in {path_to_folder}.")
        self.images = [os.path.join(path_to_folder, img) for img in os.listdir(path_to_folder)]
        if scaling_factor not in [2, 4, 8]:
            raise ValueError("Scaling factor must be either 2, 4, or 8.")
        self.scaling_factor = scaling_factor
        self.hr_resize = Resize((img_h, img_w))
        self.lr_resize = Resize((img_h // self.scaling_factor, img_w // self.scaling_factor))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        gt_image = cv2.imread(self.images[idx]).astype(np.float32) / 255.
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_tensor = image_to_tensor(gt_image, False, False)
        gt_tensor = self.hr_resize(gt_tensor)
        lr_tensor = self.lr_resize(gt_tensor)

        return lr_tensor, gt_tensor

