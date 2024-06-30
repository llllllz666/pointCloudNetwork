import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ClothSynthetic(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        self.mode = mode
        self.num_points = cfg["num_points"]
    def __len__(self):
        return 
    def __getitem__(self):
        return 