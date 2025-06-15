import os
import time
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,random_split
from skimage import color
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
from torchvision.models.segmentation import segmentation

# -----------------------------
# Section 1: Utility Functions
# -----------------------------

def srgb_to_linear(img_srgb: torch.Tensor) -> torch.Tensor:
    mask = img_srgb <= 0.04045.to(img_srgb.dtype)
    c_lin_;pw