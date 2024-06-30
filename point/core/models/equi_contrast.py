import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class EquiContrast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    