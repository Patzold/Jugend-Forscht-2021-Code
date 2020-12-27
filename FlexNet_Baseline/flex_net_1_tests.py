import os
os.chdir("flex_nets")
import random
import matplotlib.pyplot as plt
import datetime
import cv2
import pickle
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import flex_net_1 as flex

# For reproducibility
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

base_dir = "C:/Datasets/PJF-25/data/"
save_dir = "C:/Datasets/PJF-25/safe/"
categorys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]
current_categorys = ["15", "16", "17", "18", "19", "20", "0", "1", "2", "4", "5", "8", "9", "10", "11", "12"]

# img = cv2.imread(("C:/Datasets/PJF-25/data/9/comp/image0021.png"))
# img = cv2.resize(img, (224, 224))
# img_tensor = torch.from_numpy(img).to(torch.float32)

# print(flex.predict(img_tensor))

# quit()

data = []

if False:
    for cat in tqdm(current_categorys):
        # try:
        path = base_dir + cat + "/comp/"
        out = [[],[]]
        print(cat, int(cat))
        for num, img in enumerate(os.listdir(path)):
            img_in = cv2.imread((path + "/" + img), cv2.IMREAD_COLOR)
            img_resz = cv2.resize(img_in, (224, 224))
            if num < 2000:
                out[0].append([img_resz])
            else:
                out[1].append([img_resz])
        out = np.array(out)
        data.append([out, int(cat), cat])
        # except Exception as e:
        #     print("EXEPTION: ", e)
        #     print(indx, cat)
        #     print(img)
    pickle_out = open((save_dir + "cats_small.pickle"),"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
else:
    pickle_in = open(save_dir + "cats_small.pickle","rb")
    data = pickle.load(pickle_in)

data = np.array(data)
print(data.shape, data[0][0].shape, len(data[0][0][0]), len(data[0][0][1]))

train_on_gpu = torch.cuda.is_available()
theCPU = torch.device("cpu")

if not train_on_gpu:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')

train = []
test = []
lable = []

t0 = time.time()

for category in range(len(data)):
    train_total = 0
    train_correct = 0
    test_total = 0
    test_correct = 0
    images = data[category][0][0]
    correct_class = data[category][1]
    print(len(images), correct_class)
    # Train images:
    for num in range(len(images)):
        image = images[num][0]
        image_tensor = torch.from_numpy(image).to(torch.float32)
        predicted_class = flex.predict(image_tensor)
        if predicted_class == correct_class:
            train_correct += 1
        train_total += 1
    train_percentage = round(train_correct/train_total, 3)
    print("Train --> ", "Total: ", train_total, ", correct: ", train_correct, "  --> ", train_percentage, "% accuracy")
    # Test images:
    images = data[category][0][1]
    for num in range(len(images)):
        image = images[num][0]
        image_tensor = torch.from_numpy(image).to(torch.float32)
        predicted_class = flex.predict(image_tensor)
        if predicted_class == correct_class:
            test_correct += 1
        test_total += 1
    test_percentage = round(test_correct/test_total, 3)
    print("Test --> ", "Total: ", test_total, ", correct: ", test_correct, "  --> ", test_percentage, "% accuracy")
    train.append(train_percentage)
    test.append(test_percentage)
    lable.append(data[category][2])
t1 = time.time()
time_spend = t1-t0

print("Execution time:", time_spend)

x = np.arange(len(lable))  # the label locations
print("plt x:", x)
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train, width, label='In sample')
rects2 = ax.bar(x + width/2, test, width, label='Out of sample')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per class')
ax.set_xticks(x)
ax.set_xticklabels(lable, rotation="vertical")
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()
plt.savefig("FlexNet_1_1.png")
plt.show()