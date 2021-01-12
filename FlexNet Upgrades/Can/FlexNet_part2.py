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

base_dir = "C:/Datasets/PJF-30/data/"
save_dir = "C:/Datasets/PJF-30/safe/"
categorys = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [18, 19, 20, 21]]

pickle_in = open(save_dir + "fc_out.pickle","rb")
fc_out = pickle.load(pickle_in)
# predicted cat, image, real cat, real class

out_train, out_test = fc_out
print(len(out_train), len(out_test))
print(len(out_train[0]), len(out_test[0]))
print(out_train[0], out_test[0])

predicted_category, images, real_category, real_class = out_train
predicted_categoryt, imagest, real_categoryt, real_classt = out_test

print(len(predicted_category), len(images), len(real_category), len(real_class))

for i in range(len(predicted_category)):
    print(predicted_category[i], real_category[i], real_class[i])
    cv2.imshow("img", images[i])
    cv2.waitKey(0)

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

class Cans(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, 2)
        self.conv2 = nn.Conv2d(50, 100, 2)
        self.dropout = nn.Dropout(0.7)
        
        x = torch.randn(224,224,3).view(-1,3,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 500) #flattening.
        self.fc2 = nn.Linear(500, 100)
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
                print("Classes: Cans loaded")
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

cans = Cans()
cans.load_state_dict(torch.load("C:/Cache/PJF-30/classes_can_1.pt"))
cans.to(device)
cans.eval()

correct = 0
cat_correct = 0
test_cat_correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(out_test))):
        total += 1
        predicted_category = out_test[i].cpu().numpy().tolist()
        correct_category = img_in_ordert[i][1]
        correct_class = img_in_ordert[i][2]
        if predicted_category == correct_category: test_cat_correct += 1
        input_tensor = torch.from_numpy(img_in_order[i][0]).to(device).to(torch.float32).view(-1, 3, 224, 224)
        if predicted_category == 0:
            if predicted_category == correct_category: cat_correct += 1
            predicted_class = torch.argmax(rubberts(input_tensor)).cpu().numpy().tolist() + 1
        elif predicted_category == 1:
            if predicted_category == correct_category: cat_correct += 1
            predicted_class = torch.argmax(pigs(input_tensor)).cpu().numpy().tolist() + 4
        elif predicted_category == 2:
            if predicted_category == correct_category: cat_correct += 1
            predicted_class = torch.argmax(legos(input_tensor)).cpu().numpy().tolist() + 8
        elif predicted_category == 3:
            if predicted_category == correct_category: cat_correct += 1
            predicted_class = torch.argmax(cans(input_tensor)).cpu().numpy().tolist() + 14
        else:
            raise Exception("A serious problem just occoured.")
        if predicted_class == correct_class: correct += 1

print(total, correct, "  --> ", round(correct/total, 3))
print(cat_correct, test_cat_correct)