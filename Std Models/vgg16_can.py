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

if True:
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

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

vgg16 = models.vgg16(pretrained=True)

print("In features: ", vgg16.classifier[6].in_features)
print("Out features: ", vgg16.classifier[6].out_features)

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

n_inputs = vgg16.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 17)

vgg16.classifier[6] = last_layer

print(vgg16)

# torch.save(vgg16, save_dir + "pre_vgg16_model.pt")

# if GPU is available, move the model to GPU
train_on_gpu = True
if train_on_gpu:
    vgg16.cuda()

optimizer = optim.Adam(vgg16.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

BATCH_SIZE = 100
EPOCHS = 150
train_log = []
eval_size = int(len(X)*0.1)
eval_X = X[:eval_size]
eval_y = y[:eval_size]
print("After eval split: ", X.shape, y.shape)

train_data = []
log = []
valid_loss_min = np.Inf # track change in validation loss
valid_acc_min = 0

def evaluate():
    vgg16.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(eval_X))):
            real_class = eval_y[i].to(device)
            net_out = vgg16(eval_X[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
            predicted_class = torch.argmax(net_out)
            # print(real_class, net_out, predicted_class)
            # input()
            if predicted_class == real_class:
                correct += 1
            # else: cv2.imwrite(("D:/Datasets\stupid/test/o" + str(i) + ".jpg"), eval_X[i].view(75, 75, 1).numpy())
            total += 1
    in_sample_acc = round(correct/total, 3)
    correct = 0
    total = 0
    Xta = Xt[:1500]
    yta = yt[:1500]
    check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with torch.no_grad():
        for i in tqdm(range(len(Xta))):
            real_class = yta[i].to(device)
            net_out = vgg16(Xta[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
            predicted_class = torch.argmax(net_out)
            # print(real_class, net_out, predicted_class)
            # input()
            if predicted_class == real_class:
                correct += 1
                check[predicted_class.cpu().numpy()] += 1
            # else: cv2.imwrite(("D:/Datasets\stupid/test/i" + str(i) + ".jpg"), Xt[i].view(60, 60, 1).numpy())
            total += 1
    print(check)
    out_of_sample_acc = round(correct/total, 3)
    return in_sample_acc, out_of_sample_acc

t0 = time.time()
for epoch in range(EPOCHS):
    dtm = str(datetime.datetime.now())
    for i in tqdm(range(0, len(X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        vgg16.train()
        # try:
        batch_X = X[i:i+BATCH_SIZE]
        batch_y = y[i:i+BATCH_SIZE]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_data = []

        # Actual training
        vgg16.zero_grad()
        optimizer.zero_grad()
        outputs = vgg16(batch_X.view(-1, 3, 224, 224))
        # print(batch_y, outputs)
        # input()
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step() # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")
    isample, osample = evaluate()
    print("In-sample accuracy: ", isample, "  Out-of-sample accuracy: ", osample)
    train_data.append([isample, osample])
    log.append([isample, osample, loss, dtm])
    if osample > valid_acc_min and epoch > 10:
        print('Acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_acc_min, osample))
        torch.save(vgg16.state_dict(), "C:/Cache/PJF-30/std_vgg16_can.pt") #                                                  <-- UPDATE
        valid_acc_min = osample
t1 = time.time()
time_spend = t1-t0
print("Time spend: ", time_spend)
print(valid_acc_min)

train_data = np.array(train_data)
isample = train_data[:, 0]
osample = train_data[:, 1]

plt.plot(isample)
plt.plot(osample)
plt.title("Model evaluation results")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accuracy (in percentages)")
plt.legend(["in-sample", "out-of-sample"], loc="lower right")
plt.ylim([0, 1])
plt.savefig(("std_vgg16_can.pdf")) #                                              <-- UPDATE
plt.show()

# Max Out of Sample Accuracy: 0.233            4h 35min 45s