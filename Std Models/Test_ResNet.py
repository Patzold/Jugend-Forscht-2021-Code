import os
os.chdir("Std Models")
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
categorys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21]

train = []
test = []

if False:
    for indx, dir in tqdm(enumerate(categorys)):
        path = base_dir + str(dir) + "/comp/"
        out = []
        print(indx, dir)
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                if num < 2000:
                    train.append([img_resz, indx])
                else:
                    test.append([img_resz, indx])
            except Exception as e: pass
        print(len(train), len(test))

    pickle_out = open((save_dir + "vgg16_can.pickle"),"wb")
    pickle.dump(train, pickle_out)
    pickle_out.close()
    pickle_out = open((save_dir + "vgg16_cant.pickle"),"wb")
    pickle.dump(test, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "vgg16_can.pickle","rb")
    train = pickle.load(pickle_in)
    pickle_in = open(save_dir + "vgg16_cant.pickle","rb")
    test = pickle.load(pickle_in)
l = len(train)
check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(l):
    check[train[i][1]] += 1
print(check)
lt = len(test)
check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(lt):
    check[test[i][1]] += 1
print(check)
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
print(y[:10], yt[:10])
check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(l):
    check[y[i].numpy()] += 1
print(check)

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module): # [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) # 256 --> 64, 64*4 (256) again
        
        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=17):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

model = ResNet50(3, 17)
model.load_state_dict(torch.load("C:/Cache/PJF-30/std_ResNet-50_can.pt"))
model.to(device)
model.eval()

total = 0
class_correct = 0
category_correct = 0

class_check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in tqdm(range(len(X))):
    input_tensor = X[i].view(-1, 3, 224, 224).to(device)
    correct_class = y[i].cpu().numpy().tolist()
    predicted_class = torch.argmax(model(input_tensor)[0])
    if predicted_class == correct_class:
        class_correct += 1
        class_check[predicted_class-1] += 1
    total += 1

print(total, category_correct, class_correct)
print("--> ", round(category_correct / total, 3), round(class_correct / total, 3))
print(cat_check, class_check)
print("Class accuracy: ", [round(cat_check[i] / cat_total[i], 3) for i in range(len(class_total))])

# Test: 8500, 4744, 4248  --> 0.558, 0.5      (1m 45s)
#      Cat_Check: [1215, 1736, 686, 1107]     Class_Check: [416, 412, 366, 454, 429, 408, 415, 186, 105, 125, 33, 85, 45, 0, 0, 0, 0, 263, 251, 90, 165]
#      Category accuracy:  [0.81, 0.868, 0.229, 0.553]
# Train: 34000, 0 (not available), 25954  --> 0.0 0.998    (8m 06s)
#      Cat_Check: (not available)             Class_Check: [1983, 2000, 2000, 2000, 2000, 2000, 1991, 2000, 1996, 1999, 1992, 1998, 2000, 1996, 1998, 1980, 0, 0, 0, 0, 1995]
#      Category accuracy:  (not available)