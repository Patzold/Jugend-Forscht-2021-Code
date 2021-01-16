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
    integers = inpt[:4]
    floats = inpt[4:]
    cnt = integers.count(1)
    if cnt == 1: return integers.index(1)
    big = max(floats)
    return int(round((floats.index(big) / 2), 0))

def chooser_analytics(inpt):
    integers = inpt[:4]
    floats = inpt[4:]
    cnt = integers.count(1)
    if cnt == 1: return integers.index(1)
    if cnt == 0: return 0
    if cnt == 2: return 2
    else: return 3

total = 0
correct = 0

for i in range(len(yt)):
    total += 1
    if chooser(Xt[i]) == yt[i]:
        correct += 1

print("Simple:")
print(total, correct)
print(round(correct/total, 3))

total = 0
correct = 0

for i in range(len(yt)):
    total += 1
    if chooser_analytics(Xt[i]) == yt[i]:
        correct += 1

print("Analytics")
print(total, correct)
print(round(correct/total, 3))

# ANALYTICS
# none = [0, 0, 0, 0]
# one = [0, 0, 0, 0]
# two = [0, 0, 0, 0]
# three = [0, 0, 0, 0]
# four = [0, 0, 0, 0]

# for i in tqdm(range(len(y))):
#     integers = X[i][:4]
#     floats = X[i][4:]
#     cnt = integers.count(1)
#     if cnt == 0:
#         none[y[i]] += 1
#     if cnt == 1:
#         one[y[i]] += 1
#     if cnt == 2:
#         two[y[i]] += 1
#     if cnt == 3:
#         three[y[i]] += 1
#     if cnt == 4:
#         four[y[i]] += 1

# print(none, one, two, three, four)