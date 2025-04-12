import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(1024, 1024)):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.label_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]
        self.target_size = target_size
        self.label_encoder = LabelEncoder()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.target_size).astype(np.float32) / 255.0
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST).astype(np.int64)

        label_flat = label.flatten()
        encoded_label = self.label_encoder.fit_transform(label_flat).reshape(label.shape)

        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(encoded_label)
        return image, label