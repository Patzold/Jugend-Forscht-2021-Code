import os
os.chdir("Categorys v2")
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
nos = [1, 2, 3] #, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21]
yes = [4, 5, 6, 7]
all = [1, 2, 3, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21]

train = []
test = []

if True:
    out_train = []
    out_test = []
    for indx, dir in tqdm(enumerate(nos)):
        path = base_dir + str(dir) + "/comp/"
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                if num < 2000:
                    out_train.append([img_resz, np.eye(2)[0]])
                else:
                    out_test.append([img_resz, np.eye(2)[0]])
            except Exception as e: pass
    random.shuffle(out_train)
    random.shuffle(out_test)
    train += out_train[:6000]
    test += out_test[:1500]
    print(len(train), len(test), "\n")
    out_train = []
    out_test = []
    for indx, dir in tqdm(enumerate(yes)):
        path = base_dir + str(dir) + "/comp/"
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                if num < 2000:
                    out_train.append([img_resz, np.eye(2)[1]])
                else:
                    out_test.append([img_resz, np.eye(2)[1]])
            except Exception as e: pass
    random.shuffle(out_train)
    random.shuffle(out_test)
    train += out_train[:6000]
    test += out_test[:1500]
    print(len(train), len(test))
    print(len(train), len(test))

    # train = np.array(train)
    pickle_out = open((save_dir + "categorys2_pig_1.pickle"),"wb")
    pickle.dump(train, pickle_out)
    pickle_out.close()
    pickle_out = open((save_dir + "categorys2_pig_1t.pickle"),"wb")
    pickle.dump(test, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "categorys2_pig_1.pickle","rb")
    train = pickle.load(pickle_in)
    pickle_in = open(save_dir + "categorys2_pig_1t.pickle","rb")
    test = pickle.load(pickle_in)

if True:
    out_train = []
    out_test = []
    for indx, dir in tqdm(enumerate(all)):
        path = base_dir + str(dir) + "/comp/"
        for num, img in enumerate(os.listdir(path)):
            try:
                img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
                img_resz = cv2.resize(img_in, (224, 224))
                if num < 2000:
                    out_train.append([img_resz, np.eye(2)[0]])
                else:
                    out_test.append([img_resz, np.eye(2)[0]])
            except Exception as e: pass
    random.shuffle(out_train)
    random.shuffle(out_test)
    pickle_out = open((save_dir + "cat_all_pig.pickle"),"wb")
    pickle.dump(out_test, pickle_out)
    pickle_out.close()
    pickle_out = open((save_dir + "cat_all_pigt.pickle"),"wb")
    pickle.dump(out_train, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "cat_all_pig.pickle","rb")
    cat_all = pickle.load(pickle_in)
    pickle_in = open(save_dir + "cat_all_pigt.pickle","rb")
    cat_allt = pickle.load(pickle_in)
l = len(train)
lt = len(test)
print(len(train), len(test))
random.shuffle(train)
random.shuffle(test)

X, y, Xt, yt, ay, ax, ayt, axt = [], [], [], [], [], [], [], []

for features, lables in train:
    X.append(features)
    y.append(lables)
for features, lables in test:
    Xt.append(features)
    yt.append(lables)
for features, lables in cat_all:
    ax.append(features)
    ay.append(lables)
for features, lables in cat_allt:
    axt.append(features)
    ayt.append(lables)
temp = np.array(y)
# print(np.max(temp))
X = np.array(X, dtype=np.float32) / 255
y = np.array(y, dtype=np.float32)
Xt = np.array(Xt, dtype=np.float32) / 255
yt = np.array(yt, dtype=np.float32)
ax = np.array(ax, dtype=np.float32) / 255
ay = np.array(ay, dtype=np.float32)
axt = np.array(axt, dtype=np.float32) / 255
ayt = np.array(ayt, dtype=np.float32)
print(np.max(X[0]), np.max(Xt[0]))
print(y, y.shape, type(y[0]))

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X.to(torch.float32)
y.to(torch.float32)
print(X.dtype, y.dtype)
Xt = torch.from_numpy(Xt)
yt = torch.from_numpy(yt)
Xt.to(torch.float32)
yt.to(torch.float32)
print(Xt.dtype, yt.dtype)
ax = torch.from_numpy(ax)
ay = torch.from_numpy(ay)
ax.to(torch.float32)
ay.to(torch.float32)
axt = torch.from_numpy(axt)
ayt = torch.from_numpy(ayt)
axt.to(torch.float32)
ayt.to(torch.float32)
print(y[10:], yt[:10])

check = [0, 0, 0]
for i in range(l):
        check[np.argmax(y[i].numpy())] += 1
print(check)
check = [0, 0, 0]
for i in range(lt):
        check[np.argmax(yt[i].numpy())] += 1
print(check)

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

# THE NETWORK

class Net(nn.Module):
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

net = Net()
# torch.save(net, save_dir + "smple_conv_model.pt")
net.to(device)
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

BATCH_SIZE = 100
EPOCHS = 50

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
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(eval_X))):
            real_class = torch.argmax(eval_y[i].to(device))
            net_out = net(eval_X[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
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
    # Xta = Xt[:1500]
    # yta = yt[:1500]
    check = [0, 0, 0, 0, 0, 0]
    with torch.no_grad():
        for i in tqdm(range(len(Xt))):
            real_class = torch.argmax(yt[i].to(device))
            net_out = net(Xt[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
            predicted_class = torch.argmax(net_out)
            # print(real_class, net_out, predicted_class)
            # input()
            if predicted_class == real_class:
                correct += 1
                check[predicted_class.cpu().numpy()] += 1
            # else: cv2.imwrite(("D:/Datasets\stupid/test/i" + str(i) + ".jpg"), Xt[i].view(60, 60, 1).numpy())
            total += 1
    print(check, "--------------------------")
    with torch.no_grad():
        total = 0
        correct = 0
        check = [0, 0]
        realcheck = [0, 0]
        with torch.no_grad():
            for i in tqdm(range(len(ay))):
                real_class = torch.argmax(ay[i].to(device))
                realcheck[real_class] += 1
                net_out = net(ax[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
                predicted_class = torch.argmax(net_out)
                check[predicted_class] += 1
                if predicted_class == real_class: correct += 1
                total += 1
        print("All: ", check, realcheck, total, correct)
        total = 0
        correct = 0
        check = [0, 0]
        realcheck = [0, 0]
        with torch.no_grad():
            for i in tqdm(range(len(ayt))):
                real_class = torch.argmax(ayt[i].to(device))
                realcheck[real_class] += 1
                net_out = net(axt[i].view(-1, 3, 224, 224).to(device))[0]  # returns a list
                predicted_class = torch.argmax(net_out)
                check[predicted_class] += 1
                if predicted_class == real_class: correct += 1
                total += 1
        print("All test: ", check, realcheck, total, correct)    
    out_of_sample_acc = round(correct/total, 3)
    return in_sample_acc, out_of_sample_acc

t0 = time.time()
for epoch in range(EPOCHS):
    dtm = str(datetime.datetime.now())
    for i in tqdm(range(0, len(X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        net.train()
        # try:
        batch_X = X[i:i+BATCH_SIZE]
        batch_y = y[i:i+BATCH_SIZE]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_data = []

        # Actual training
        net.zero_grad()
        optimizer.zero_grad()
        outputs = net(batch_X.view(-1, 3, 224, 224))
        # print(batch_y.type(), outputs.type())
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
        # torch.save(net.state_dict(), "C:/Cache/PJF-30/categorys2_pig_1.pt") #                                                  <-- UPDATE
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
# plt.savefig(("2pig_1.pdf")) #                                              <-- UPDATE
plt.show()

# Time spend: 6m 21s
# In-sample: 99,0%   Out-of-sample: 98,0%