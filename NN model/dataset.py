import torch
from torch.utils.data import Dataset
import numpy as np
import os
from features import extract_features

class StrokeDataset(Dataset):
    def __init__(self, root_dir):
        self.stroke_dir = os.path.join(root_dir, "strokes")
        self.label_dir = os.path.join(root_dir, "labels")
        self.files = sorted(os.listdir(self.stroke_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        stroke = np.load(os.path.join(self.stroke_dir, name))
        labels = np.load(os.path.join(self.label_dir, name))

        features = extract_features(stroke)

        return (
            torch.tensor(features, dtype=torch.float32),  # (N, F)
            torch.tensor(labels, dtype=torch.long)        # (N,)
        )
