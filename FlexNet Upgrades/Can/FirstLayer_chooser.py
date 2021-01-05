import os
os.chdir("FlexNet Upgrades/Can")
import random
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import pickle
from tqdm import tqdm
import numpy as np

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

pickle_in = open(save_dir + "can_intm_3_raw.pickle","rb")
train = pickle.load(pickle_in)
pickle_in = open(save_dir + "can_intm_3t_raw.pickle","rb")
test = pickle.load(pickle_in)

l = len(train)
lt = len(test)
print(len(train), len(test))
random.shuffle(train)
random.shuffle(test)

X, y, Xt, yt = [],  [], [],  []

for features, lables in train:
    X.append(features)
    y.append(lables)
for features, lables in test:
    Xt.append(features)
    yt.append(lables)
temp = np.array(y)
print(np.max(temp))

check = [0, 0, 0, 0]
for i in range(l):
        check[y[i]] += 1
print(check)
check = [0, 0, 0, 0]
for i in range(lt):
        check[yt[i]] += 1
print(check)

def chooser(inpt):
    integers = y[:4]
    floats = y[3:]
    print(integers, floats)

print(len(X), len(y))
print(len(X), len(y))
input()

chooser(X[0])
print(y[0])