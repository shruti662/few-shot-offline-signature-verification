import os
import cv2
import random
import torch
from torch.utils.data import Dataset


class SiameseSignatureDataset(Dataset):
    def __init__(self, genuine_root, forged_root):
        """
        genuine_root → dataset/DATASET/train/genuine
        forged_root  → dataset/DATASET/test/forged
        """

        self.genuine_users = {}
        self.forged_users = {}

        # -------- Load Genuine Per User --------
        for user in os.listdir(genuine_root):
            user_path = os.path.join(genuine_root, user)

            if os.path.isdir(user_path):
                images = [
                    os.path.join(user_path, f)
                    for f in os.listdir(user_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                if len(images) > 0:
                    self.genuine_users[user] = images

        # -------- Load Forged Per User --------
        for user in os.listdir(forged_root):
            user_path = os.path.join(forged_root, user)

            if os.path.isdir(user_path):
                images = [
                    os.path.join(user_path, f)
                    for f in os.listdir(user_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                if len(images) > 0:
                    self.forged_users[user] = images

        self.users = list(self.genuine_users.keys())

        if len(self.users) == 0:
            raise ValueError("No genuine user folders found.")

    def __len__(self):
        # Define number of pairs per epoch
        return len(self.users) * 10  # adjustable for speed

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load {path}")

        img = cv2.resize(img, (160, 120))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img

    def __getitem__(self, idx):
        # Randomly select a user
        user = random.choice(self.users)

        genuine_images = self.genuine_users[user]

        # 50% Positive pair (Genuine-Genuine)
        if random.random() < 0.5 and len(genuine_images) > 1:

            img1_path, img2_path = random.sample(genuine_images, 2)

            label = 1  # Genuine pair

        # 50% Negative pair (Genuine-Forgery same writer)
        else:

            if user in self.forged_users and len(self.forged_users[user]) > 0:
                img1_path = random.choice(genuine_images)
                img2_path = random.choice(self.forged_users[user])
                label = 0  # Forged pair
            else:
                # fallback if no forged available
                img1_path, img2_path = random.sample(genuine_images, 2)
                label = 1

        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        return img1, img2, torch.tensor(label, dtype=torch.float32)