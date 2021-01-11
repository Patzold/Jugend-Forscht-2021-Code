import os
os.chdir("FlexNet Upgrades/Lego")
import random
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import pickle
from tqdm import tqdm
import numpy as np

import FirstLayer_convs as fl

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

base_dir = "C:/Datasets/PJF-30/data/"
save_dir = "C:/Datasets/PJF-30/safe/"
categorys = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [18, 19, 20, 21]]

pickle_in = open(save_dir + "can_intm_3_img.pickle","rb")
img_in_order = pickle.load(pickle_in)

print(len(img_in_order))

for i in range(len(img_in_order)):
    print(img_in_order[i])
    input()