import os
os.chdir("FlexNet 2")
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

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

class Preconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 32, 2)

    def forward(self, x):
        c1 = self.conv1(x)
        print(c1.shape)
        relu1 = F.relu(c1)
        print(relu1.shape)
        maxpool1 = F.max_pool2d(relu1, (2, 2))
        return maxpool1
preconv, Preconv()
preconv.to(device)

vgg16 = models.vgg16(pretrained=True)

print("In features: ", vgg16.classifier[6].in_features)
print("Out features: ", vgg16.classifier[6].out_features)

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

print(vgg16, preconv)

layers = list(vgg16.features.children())[:-1]

first_layer = layers[0]
second_layer = layers[2]
preconv.conv1 = first_layer
preconv.conv2 = second_layer
print(preconv)

base_dir = "C:/Datasets/PJF-30/data/"
save_dir = "C:/Datasets/PJF-30/safe/"
nos = [4, 5, 6, 7] # Pig Head
yes = [1, 2, 3]

train = []
test = []

if True:
    out_train = []
    out_test = []
    for indx, dir in tqdm(enumerate(nos)):
        path = base_dir + str(dir) + "/comp/"
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                img_tnsr = torch.from_numpy(img_resz).to(torch.float32).to(device)
                net_out = preconv(img_tensor.view(-1, 3, 224, 224))
                if num < 2000:
                    out_train.append([net_out.cpu(), 0])
                    print(net_out.cpu(), net_out.cpu().shape())
                    input()
                else:
                    out_test.append([net_out.cpu(), 0])
            except Exception as e: pass
    random.shuffle(out_train)
    random.shuffle(out_test)
    train += out_train[:6000]
    test += out_test[:1500]
    print(len(train), len(test), "\n")
    out_train = []
    out_test = []
    for indx, dir in tqdm(enumerate(yes)):
        path = base_dir + str(dir) + "/comp/"
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                img_tnsr = torch.from_numpy(img_resz).to(torch.float32).to(device)
                net_out = preconv(img_tensor.view(-1, 3, 224, 224))
                if num < 2000:
                    out_train.append([net_out.cpu(), 1])
                else:
                    out_test.append([net_out.cpu(), 1])
            except Exception as e: pass
    random.shuffle(out_train)
    random.shuffle(out_test)
    train += out_train[:6000]
    test += out_test[:1500]
    print(len(train), len(test))
    print(len(train), len(test))

    # train = np.array(train)
    pickle_out = open((save_dir + "rnd_cat_rubt_1.pickle"),"wb")
    pickle.dump(train, pickle_out)
    pickle_out.close()
    pickle_out = open((save_dir + "rnd_cat_rubt_1t.pickle"),"wb")
    pickle.dump(test, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "rnd_cat_rubt_1.pickle","rb")
    train = pickle.load(pickle_in)
    pickle_in = open(save_dir + "rnd_cat_rubt_1t.pickle","rb")
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
X = np.array(X, dtype=np.float32) / 255
y = np.array(y, dtype=np.int64)
Xt = np.array(Xt, dtype=np.float32) / 255
yt = np.array(yt, dtype=np.int64)
print(np.max(X[0]), np.max(Xt[0]))

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X.to(torch.float32)
y.to(torch.int64)
print(X.dtype, y.dtype)
Xt = torch.from_numpy(Xt)
yt = torch.from_numpy(yt)
Xt.to(torch.float32)
yt.to(torch.int64)
print(Xt.dtype, yt.dtype)
print(y[10:], yt[:10])

check = [0, 0, 0]
for i in range(l):
        check[y[i].numpy()] += 1
print(check)
check = [0, 0, 0]
for i in range(lt):
        check[yt[i].numpy()] += 1
print(check)