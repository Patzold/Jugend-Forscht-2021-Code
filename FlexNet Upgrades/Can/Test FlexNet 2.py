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

import FlexNet_part1 as flex

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
# categorys = [18, 19, 20, 21]
categorys = [8, 9, 10, 11, 12, 13]
the_category = 2

train = []
test = []

if True:
    for dir in tqdm(categorys):
        out_train = []
        out_test = []
        actual_class = dir
        # if actual_class > 13:
        #     actual_class = actual_class - 4
        path = base_dir + str(dir) + "/comp/"
        print(dir, actual_class)
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                if num < 2000:
                    out_train.append([img_resz, the_category, actual_class])
                else:
                    out_test.append([img_resz, the_category, actual_class])
            except Exception as e: pass
        random.shuffle(train)
        random.shuffle(test)
        train += out_train #[:6000]
        test += out_test #[:1500]
    print(len(train), len(test))

    # train = np.array(train)
#     pickle_out = open((save_dir + "test_can.pickle"),"wb")
#     pickle.dump(train, pickle_out)
#     pickle_out.close()
#     pickle_out = open((save_dir + "test_can_t.pickle"),"wb")
#     pickle.dump(test, pickle_out)
#     pickle_out.close()
# else:
#     pickle_in = open(save_dir + "test_can.pickle","rb")
#     train = pickle.load(pickle_in)
#     pickle_in = open(save_dir + "test_can_t.pickle","rb")
#     test = pickle.load(pickle_in)
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

X, y, c, Xt, yt, ct = [],  [], [],  [], [], []

for features, cat, clas in train:
    X.append(features)
    y.append(clas)
    c.append(cat)
for features, cat, clas in test:
    Xt.append(features)
    yt.append(clas)
    ct.append(cat)
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

total = 0
class_correct = 0
category_correct = 0

cat_check = [0, 0, 0, 0]
cat_total = [0, 0, 0, 0]

for i in tqdm(range(len(Xt))):
    input_tensor = Xt[i].view(-1, 3, 224, 224).to(device)
    correct_class = yt[i].cpu().numpy().tolist()
    predicted_category = flex.predict(input_tensor)
    if predicted_category == ct[i]:
        category_correct += 1
        cat_check[predicted_category] += 1
    cat_total[ct[i]] += 1
    total += 1

print("--> ", round(category_correct / total, 3))
print(cat_check, cat_total)
print("Category accuracy: ", [round(cat_check[i] / cat_total[i], 3) for i in range(len(cat_total))])