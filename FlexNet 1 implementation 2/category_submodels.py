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

pickle_in = open(save_dir + "img_tensor.pickle","rb")
train = pickle.load(pickle_in)
pickle_in = open(save_dir + "img_tensor_test.pickle","rb")
test = pickle.load(pickle_in)

X, y = train
Xt, yt = test

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

class RubberToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.dropout = nn.Dropout(0.75)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 300) #flattening.
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 2)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("to linear: ", self._to_linear)
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class PigHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.dropout = nn.Dropout(0.75)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 300) #flattening.
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 2)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("to linear: ", self._to_linear)
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Lego(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.dropout = nn.Dropout(0.75)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 300) #flattening.
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 2)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("to linear: ", self._to_linear)
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Can(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.dropout = nn.Dropout(0.75)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 500) #flattening.
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def convs(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
                print("to linear: ", self._to_linear)
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

rubt, pig, lego, can = RubberToy(), PigHead(), Lego(), Can()
rubt.load_state_dict(torch.load("C:/Cache/PJF-30/categorys2_rubt_1.pt"))
pig.load_state_dict(torch.load("C:/Cache/PJF-30/categorys2_pig_1.pt"))
lego.load_state_dict(torch.load("C:/Cache/PJF-30/categorys2_lego_1.pt"))
can.load_state_dict(torch.load("C:/Cache/PJF-30/categorys2_can_1.pt"))
rubt.to(device)
pig.to(device)
lego.to(device)
can.to(device)
rubt.eval()
pig.eval()
lego.eval()
can.eval()

# v1: raw net output
# v2: argmax
# v3: raw output & argmax
check_integers = [0, 0, 0, 0]
def run(input_tensor):
    with torch.no_grad():
        rubt_out = rubt(input_tensor).cpu().numpy().tolist()[0]
        rubt_argmax = torch.argmax(rubt(input_tensor)).cpu().numpy().tolist()
        if rubt_argmax == 1: check_integers[0] += 1
        pig_out = pig(input_tensor).cpu().numpy().tolist()[0]
        pig_argmax = torch.argmax(pig(input_tensor).cpu()).numpy().tolist()
        if pig_argmax == 1: check_integers[1] += 1
        lego_out = lego(input_tensor).cpu().numpy().tolist()[0]
        lego_argmax = torch.argmax(lego(input_tensor).cpu()).numpy().tolist()
        if lego_argmax == 1: check_integers[2] += 1
        can_out = can(input_tensor).cpu().numpy().tolist()[0]
        can_argmax = torch.argmax(can(input_tensor).cpu()).numpy().tolist()
        if can_argmax == 1: check_integers[3] += 1
        out = [rubt_argmax, pig_argmax, lego_argmax, can_argmax] + rubt_out + pig_out + lego_out + can_out
        # out = [rubt_argmax, pig_argmax]  # v2
        # out = rubt_out + pig_out  # v1
        return out

intm = []
analytics = [0, 0, 0, 0, 0]
for i in tqdm(range(len(y))):
    result = run(X[i].view(-1, 3, 224, 224).to(device))
    integers = result[:4]
    count = integers.count(1)
    analytics[count] += 1
    intm.append(result)
print("Analytics: ", analytics)
print("Check integers: ", check_integers)
quit()
pickle_out = open((save_dir + "intm.pickle"),"wb")
pickle.dump(intm, pickle_out)
pickle_out.close()

intm = []
for i in tqdm(range(len(yt))):
    result = run(Xt[i].view(-1, 3, 224, 224).to(device))
    intm.append(result)
pickle_out = open((save_dir + "intm_t.pickle"),"wb")
pickle.dump(intm, pickle_out)
pickle_out.close()

# All 17
# Train part: 2m 57s
# Test part: 44s

# Baseline
# Train part: 1m 9s
# Test part: 16s