from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class VanGoghPhotoDataset(Dataset):
    def __init__(self, root_vangogh, root_photo, transform=None):
        self.root_vangogh = root_vangogh  # e.g., "dataset/trainA"
        self.root_photo = root_photo      # e.g., "dataset/trainB"
        self.transform = transform

        self.vangogh_images = os.listdir(root_vangogh)
        self.photo_images = os.listdir(root_photo)
        self.length_dataset = max(len(self.vangogh_images), len(self.photo_images))
        self.vangogh_len = len(self.vangogh_images)
        self.photo_len = len(self.photo_images)

        # Preload images if RAM allows (optional for RTX 3050 with limited memory)
        self.preloaded = False
        if os.path.exists(root_vangogh) and os.path.exists(root_photo):
            self.vangogh_data = [np.array(Image.open(os.path.join(root_vangogh, img)).convert("RGB")) for img in self.vangogh_images]
            self.photo_data = [np.array(Image.open(os.path.join(root_photo, img)).convert("RGB")) for img in self.photo_images]
            self.preloaded = True

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        vangogh_idx = index % self.vangogh_len
        photo_idx = index % self.photo_len

        if self.preloaded:
            vangogh_img = self.vangogh_data[vangogh_idx]
            photo_img = self.photo_data[photo_idx]
        else:
            vangogh_path = os.path.join(self.root_vangogh, self.vangogh_images[vangogh_idx])
            photo_path = os.path.join(self.root_photo, self.photo_images[photo_idx])
            vangogh_img = np.array(Image.open(vangogh_path).convert("RGB"))
            photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=vangogh_img, image0=photo_img)
            vangogh_img = augmentations["image"]
            photo_img = augmentations["image0"]

        return vangogh_img, photo_img
