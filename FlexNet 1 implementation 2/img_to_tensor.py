import os
os.chdir("FlexNet 1 implementation 2")
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
dirct = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21]

X, y, Xt, yt = [],  [], [],  []

for indx, dir in tqdm(enumerate(dirct)):
    path = base_dir + str(dir) + "/comp/"
    for num, img in enumerate(os.listdir(path)):
        try:
            img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
            img_resz = cv2.resize(img_in, (224, 224))
            img_tnsr = torch.from_numpy(img_resz).to(torch.float32)
            class_tnsr = torch.from_numpy(np.array(indx)).to(torch.float32)
            if num < 2000:
                X.append(img_tnsr)
                y.append(class_tnsr)
            else:
                Xt.append(img_tnsr)
                yt.append(class_tnsr)
        except Exception as e: pass
print(len(X), len(y), len(Xt), len(yt))

pickle_out = open((save_dir + "img_tensor.pickle"),"wb")
pickle.dump([X, y], pickle_out)
pickle_out.close()
pickle_out = open((save_dir + "img_tensor_test.pickle"),"wb")
pickle.dump([Xt, yt], pickle_out)
pickle_out.close()

pickle_in = open(save_dir + "img_tensor.pickle","rb")
train = pickle.load(pickle_in)
pickle_in = open(save_dir + "img_tensor_test.pickle","rb")
test = pickle.load(pickle_in)

X, y = train
Xt, yt = test

print(type(X[0]), type(y[0]))
print(X[0].shape, y[0].shape)
print(y[0], y[10000])
print(y[0].numpy().tolist(), y[10000].numpy().tolist())

check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(y)):
        check[int(y[i].numpy().tolist())] += 1
print(check)
check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(yt)):
        check[int(yt[i].numpy().tolist())] += 1
print(check)