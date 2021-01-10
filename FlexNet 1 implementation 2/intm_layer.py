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

pickle_in = open(save_dir + "intm.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open(save_dir + "intm_t.pickle","rb")
Xt = pickle.load(pickle_in)

pickle_in = open(save_dir + "img_tensor.pickle","rb")
train = pickle.load(pickle_in)
pickle_in = open(save_dir + "img_tensor_test.pickle","rb")
test = pickle.load(pickle_in)
i, y = train
it, yt = test

print("X:", type(X), len(X), X[0])
print("Xt:", type(Xt), len(Xt), Xt[0])
print("y:", type(y), len(y), y[0])
print("yt:", type(yt), len(yt), yt[0])

to_shuffle_train = [[X[i], y[i]] for i in range(len(y))]
to_shuffle_test = [[Xt[i], yt[i]] for i in range(len(yt))]

random.shuffle(to_shuffle_train)
random.shuffle(to_shuffle_test)

print(len(to_shuffle_train), len(to_shuffle_train[0]), to_shuffle_train[0])

X, y, Xt, yt = [],  [], [],  []

for features, lables in to_shuffle_train:
    X.append(features)
    y_append = lables
    if lables.numpy() > 2: y_append = torch.from_numpy(np.array(1))
    else: y_append = torch.from_numpy(np.array(0))
    y.append(y_append)

for features, lables in to_shuffle_test:
    Xt.append(features)
    y_append = lables
    if lables.numpy() > 2: y_append = torch.from_numpy(np.array(1))
    else: y_append = torch.from_numpy(np.array(0))
    yt.append(y_append)

print(X[:3], y[:3])
print(Xt[:3], yt[:3])

print("X:", type(X), len(X), X[0])
print("Xt:", type(Xt), len(Xt), Xt[0])
print("y:", type(y), len(y), y[0])
print("yt:", type(yt), len(yt), yt[0])

print(y[0].numpy())
print(int(y[0].numpy()))

check = [0, 0, 0, 0]
for i in range(len(y)):
    check[int(y[i].numpy())] += 1
print(check)
check = [0, 0, 0, 0]
for i in range(len(yt)):
    check[yt[i].numpy()] += 1
print(check)

X = torch.from_numpy(np.array(X)).to(torch.float32)
Xt = torch.from_numpy(np.array(Xt)).to(torch.float32)
y = torch.from_numpy(np.array(y)).to(torch.int64)
yt = torch.from_numpy(np.array(yt)).to(torch.int64)

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

# THE NETWORK
classes = 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear((classes + classes*2), 24)
        self.fc2 = nn.Linear(24, 8)
        self.fc3 = nn.Linear(8, classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = Net()
net.to(device)
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()

BATCH_SIZE = 100
EPOCHS = 100

train_log = []
eval_size = int(len(X)*0.1)
eval_X = X[:eval_size]
eval_y = y[:eval_size]
print("After eval split: ", len(X), len(y))

train_data = []
log = []
valid_loss_min = np.Inf
valid_acc_min = 0

def evaluate():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(eval_X))):
            real_class = eval_y[i].to(device)
            net_out = net(eval_X[i].view(-1, (classes + classes*2)).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    in_sample_acc = round(correct/total, 3)
    correct = 0
    total = 0
    check = [0, 0, 0, 0]
    with torch.no_grad():
        for i in tqdm(range(len(Xt))):
            real_class = yt[i].to(device)
            net_out = net(Xt[i].view(-1, (classes + classes*2)).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
                check[predicted_class.cpu().numpy()] += 1
            total += 1
    out_of_sample_acc = round(correct/total, 3)
    print(check)
    return in_sample_acc, out_of_sample_acc

t0 = time.time()
for epoch in range(EPOCHS):
    dtm = str(datetime.datetime.now())
    for i in tqdm(range(0, len(X), BATCH_SIZE)):
        net.train()
        batch_X = X[i:i+BATCH_SIZE]
        batch_y = y[i:i+BATCH_SIZE]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_data = []

        # Actual training
        net.zero_grad()
        optimizer.zero_grad()
        outputs = net(batch_X.view(-1, (classes + classes*2)))
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}. Loss: {loss}")
    isample, osample = evaluate()
    print("In-sample accuracy: ", isample, "  Out-of-sample accuracy: ", osample)
    train_data.append([isample, osample])
    log.append([isample, osample, loss, dtm])
    if osample > valid_acc_min and epoch > 10:
        print('Acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_acc_min, osample))
        torch.save(net.state_dict(), "C:/Cache/PJF-30/intm.pt") #                                                  <-- UPDATE
        valid_acc_min = osample
t1 = time.time()
time_spend = t1-t0

print("Time spend:", time_spend)
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
plt.savefig(("intm.pdf")) #                                              <-- UPDATE
plt.show()

# Max Out of Sample Accuracy: 0.783    5min 3s         12 - 24 - 8 - 4        <-- Selected