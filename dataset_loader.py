import os
import cv2
import torch
from torch.utils.data import Dataset

class SignatureDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []

        # If folder contains images directly
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)

            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(file_path)

        # If no images found, check nested structure
        if len(self.image_paths) == 0:
            for sub in os.listdir(root_dir):
                sub_path = os.path.join(root_dir, sub)

                if os.path.isdir(sub_path):
                    for file in os.listdir(sub_path):
                        file_path = os.path.join(sub_path, file)
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(file_path)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load {img_path}")

        image = cv2.resize(image, (160, 120))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return image
    
