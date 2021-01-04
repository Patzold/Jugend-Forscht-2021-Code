import os
# os.chdir("FlexNet 2/")
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

img_in = cv2.imread("C:/Datasets/PJF-30/data/1/comp/image0087.png", cv2.IMREAD_COLOR)
img_resz = cv2.resize(img_in, (224, 224))
img_tensor = torch.from_numpy(img_resz).to(torch.float32)

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
preconv, preconv2 = Preconv(), Preconv()
# net.to(device)

vgg16 = models.vgg16(pretrained=True)

print("In features: ", vgg16.classifier[6].in_features)
print("Out features: ", vgg16.classifier[6].out_features)

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

print(vgg16, preconv2)

layers = list(vgg16.features.children())[:-1]

first_layer = layers[0]
second_layer = layers[2]
preconv2.conv1 = first_layer
preconv2.conv2 = second_layer
print(preconv2)

def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(10, 8))
    
    for i in range(n_filters):
        ax = fig.add_subplot(4, n_filters/4, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
    plt.show()

with torch.no_grad():
    net_out = preconv(img_tensor.view(-1, 3, 224, 224))
    print(net_out.shape)
    viz_layer(net_out, 32)
    net_out = preconv2(img_tensor.view(-1, 3, 224, 224))
    print(net_out.shape)
    viz_layer(net_out, 64)
    # viz_layer(first_layer)