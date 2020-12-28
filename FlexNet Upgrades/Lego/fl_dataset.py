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
categorys = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]

train = []
test = []

if False:
    for indx, cat in tqdm(enumerate(categorys)):
        out_train = []
        out_test = []
        for subindx, dir in tqdm(enumerate(cat)):
            path = base_dir + str(dir) + "/comp/"
            for num, img in enumerate(os.listdir(path)):
                try:
                    img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                    img_resz = cv2.resize(img_in, (224, 224))
                    if num < 2000:
                        out_train.append([img_resz, indx, dir])
                    else:
                        out_test.append([img_resz, indx, dir])
                except Exception as e: pass
        random.shuffle(train)
        random.shuffle(test)
        train += out_train[:6000]
        test += out_test[:1500]
    print(len(train), len(test))

    # train = np.array(train)
    pickle_out = open((save_dir + "fl_lego.pickle"),"wb")
    pickle.dump(train, pickle_out)
    pickle_out.close()
    pickle_out = open((save_dir + "fl_lego_t.pickle"),"wb")
    pickle.dump(test, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "fl_lego.pickle","rb")
    train = pickle.load(pickle_in)
    pickle_in = open(save_dir + "fl_lego_t.pickle","rb")
    test = pickle.load(pickle_in)
l = len(train)
lt = len(test)
print(len(train), len(test))
random.shuffle(train)
random.shuffle(test)

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

X, y, Xt, yt = [],  [], [],  []

for features, lables, subcat in train:
    X.append(features)
    y.append(lables)
for features, lables, subcat in test:
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

print(fl.run(X[0].view(-1, 3, 224, 224).to(device)))

intm = []

for i in tqdm(range(len(y))):
    result = fl.run(X[i].view(-1, 3, 224, 224).to(device))
    intm.append([result, y[i].cpu().numpy().tolist()])

pickle_out = open((save_dir + "lego_intm_2_raw.pickle"),"wb")
pickle.dump(intm, pickle_out)
pickle_out.close()

# intm 1t  34s
# intm 2t  35s
# intm 3t  35s

# intm 1  2m 20s
# intm 2  2m 26s
# intm 3  2m 21s