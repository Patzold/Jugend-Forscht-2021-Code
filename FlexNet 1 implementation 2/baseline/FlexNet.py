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
                print("Category: RubberToy loaded")
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
                print("Category: PigHead loaded")
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


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
                print("Category: Lego loaded")
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

rubt, pig = RubberToy(), PigHead()
rubt.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_rubt_1_1.pt"))
pig.load_state_dict(torch.load("C:/Cache/PJF-30/categorys_pig_1.pt"))
rubt.to(device)
pig.to(device)
rubt.eval()
pig.eval()

# v1: raw net output
# v2: argmax
# v3: raw output & argmax

def create_intm(input_tensor):
    with torch.no_grad():
        rubt_out = rubt(input_tensor).cpu().numpy().tolist()[0]
        rubt_argmax = torch.argmax(rubt(input_tensor)).cpu().numpy().tolist()
        pig_out = pig(input_tensor).cpu().numpy().tolist()[0]
        pig_argmax = torch.argmax(pig(input_tensor).cpu()).numpy().tolist()
        out = [rubt_argmax, pig_argmax] + rubt_out + pig_out  # v3
        return out

class FC3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 24)
        self.fc2 = nn.Linear(24, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
fc3 = FC3()
fc3.load_state_dict(torch.load("C:/Cache/PJF-30/lego_intm_3.pt"))
fc3.to(device)
fc3.eval()

predicted_category = 0

class RubberToys(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 2)
        self.conv2 = nn.Conv2d(12, 24, 2)
        self.dropout = nn.Dropout(0.5)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 200) #flattening.
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("Classes: RubberToys loaded")
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Pigs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 2)
        self.conv2 = nn.Conv2d(12, 24, 2)
        self.dropout = nn.Dropout(0.3)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 200) #flattening.
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 4)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("Classes: Pigs loaded")
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Legos(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 2)
        self.conv2 = nn.Conv2d(12, 24, 2)
        self.dropout = nn.Dropout(0.5)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 200) #flattening.
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 6)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
                print("Classes: Legos loaded")
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

rubberts = RubberToys()
rubberts.load_state_dict(torch.load("C:/Cache/PJF-30/classes_rubt_1_1.pt"))
rubberts.to(device)
rubberts.eval()

pigs = Pigs()
pigs.load_state_dict(torch.load("C:/Cache/PJF-30/classes_pig_1.pt"))
pigs.to(device)
pigs.eval()

legos = Legos()
legos.load_state_dict(torch.load("C:/Cache/PJF-30/classes_lego_1.pt"))
legos.to(device)
legos.eval()

predicted_class = 0

def predict(input_tensor):
    with torch.no_grad():
        intm = torch.from_numpy(np.array(create_intm(input_tensor))).to(torch.float32).to(device)
        predicted_category = torch.argmax(fc3(intm)).cpu().numpy().tolist()
        if predicted_category == 0:
            predicted_class = torch.argmax(rubberts(input_tensor)).cpu().numpy().tolist() + 1
            return predicted_category, predicted_class
        elif predicted_category == 1:
            predicted_class = torch.argmax(pigs(input_tensor)).cpu().numpy().tolist() + 4
            return predicted_category, predicted_class
        elif predicted_category == 2:
            predicted_class = torch.argmax(legos(input_tensor)).cpu().numpy().tolist() + 8
            return predicted_category, predicted_class
        else:
            raise Exception("A serious problem just occoured.")