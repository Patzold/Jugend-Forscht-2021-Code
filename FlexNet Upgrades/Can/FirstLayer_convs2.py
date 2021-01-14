import os
# os.chdir("FlexNet/")
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
        self.conv1 = nn.Conv2d(3, 42, 2)
        self.conv2 = nn.Conv2d(42, 84, 2)
        self.dropout = nn.Dropout(0.75)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 200) #flattening.
        self.fc2 = nn.Linear(200, 100)
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
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.dropout = nn.Dropout(0.8)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 700) #flattening.
        self.fc2 = nn.Linear(700, 100)
        self.fc3 = nn.Linear(100, 2)

    def convs(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv3(x)
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
rubt.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_rubt_1_1.pt"))
pig.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_pig_1.pt"))
lego.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_lego_2.pt"))
can.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_can_1.pt"))
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

def run(input_tensor):
    with torch.no_grad():
        rubt_out = rubt(input_tensor).cpu().numpy().tolist()[0]
        rubt_argmax = torch.argmax(rubt(input_tensor)).cpu().numpy().tolist()
        pig_out = pig(input_tensor).cpu().numpy().tolist()[0]
        pig_argmax = torch.argmax(pig(input_tensor).cpu()).numpy().tolist()
        lego_out = lego(input_tensor).cpu().numpy().tolist()[0]
        lego_argmax = torch.argmax(lego(input_tensor).cpu()).numpy().tolist()
        can_out = can(input_tensor).cpu().numpy().tolist()[0]
        can_argmax = torch.argmax(can(input_tensor).cpu()).numpy().tolist()
        out = [rubt_argmax, pig_argmax, lego_argmax, can_argmax] + rubt_out + pig_out + lego_out + can_out # v3, all
        return out