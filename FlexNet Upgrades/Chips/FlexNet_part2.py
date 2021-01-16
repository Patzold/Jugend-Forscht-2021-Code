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
# print(out_train[0], out_test[0])

predicted_category, images, real_category, real_class, predicted_categoryt, imagest, real_categoryt, real_classt = [], [], [], [], [], [], [], []

for pc, i, rct, rcl in out_train:
    predicted_category.append(pc)
    images.append(i)
    real_category.append(rct)
    real_class.append(rcl)
for pc, i, rct, rcl in out_test:
    predicted_categoryt.append(pc)
    imagest.append(i)
    real_categoryt.append(rct)
    real_classt.append(rcl)

print(len(predicted_category), len(images), len(real_category), len(real_class))
correct = 0
total = 0
for i in range(len(predicted_category)):
    total += 1
    if predicted_category[i] == real_category[i]:
        correct += 1

print(total, correct, round(correct/total, 3))

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
total = 0
cat_total = [0, 0, 0, 0]
check = [0, 0, 0, 0]
check_correct = [0, 0, 0, 0]
check_wrong = [0, 0, 0, 0]
class_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class_wrong = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
with torch.no_grad():
    for i in tqdm(range(len(predicted_categoryt))):
        total += 1
        pc = predicted_categoryt[i].cpu().numpy().tolist()
        correct_category = real_categoryt[i]
        correct_class = real_classt[i]
        input_tensor = torch.from_numpy(imagest[i]).to(device).to(torch.float32).view(-1, 3, 224, 224)
        if pc == 0:
            check[0] += 1
            if pc == correct_category:
                cat_correct += 1
                check_correct[0] += 1
            else:
                check_wrong[0] += 1
            predicted_class = torch.argmax(rubberts(input_tensor)).cpu().numpy().tolist() + 1
        elif pc == 1:
            check[1] += 1
            if pc == correct_category:
                cat_correct += 1
                check_correct[1] += 1
            else:
                check_wrong[1] += 1
            predicted_class = torch.argmax(pigs(input_tensor)).cpu().numpy().tolist() + 4
        elif pc == 2:
            check[2] += 1
            if pc == correct_category:
                cat_correct += 1
                check_correct[2] += 1
            else:
                check_wrong[2] += 1
            predicted_class = torch.argmax(legos(input_tensor)).cpu().numpy().tolist() + 8
        elif pc == 3:
            check[3] += 1
            if pc == correct_category:
                cat_correct += 1
                check_correct[3] += 1
            else:
                check_wrong[3] += 1
            predicted_class = torch.argmax(cans(input_tensor)).cpu().numpy().tolist() + 14
        else:
            raise Exception("A serious problem just occoured.")
        if predicted_class == correct_class:
            correct += 1
            class_correct[correct_class-1] += 1
        else: class_wrong[predicted_class-1] += 1
        cat_total[correct_category] += 1
        class_total[correct_class-1] += 1
class_acc = [round(class_correct[i] / class_total[i], 3) for i in range(len(class_total))]
cat_acc = [round(check_correct[i] / cat_total[i], 3) for i in range(len(cat_total))]
print(total, correct, "  --> ", round(correct/total, 3))
print(total, cat_correct, "  --> ", round(cat_correct/total, 3))

print(check, check_correct, check_wrong)
print(class_correct, class_wrong)
print("----")
print(class_acc)
print(cat_acc)