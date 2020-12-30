import os
os.chdir("visualisations")
import matplotlib.pyplot as plt
import numpy as np

lables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# names = []

baseline_val = [443, 449, 407, 481, 434, 473, 477]
lego_val = [420, 446, 375, 453, 422, 406, 414, 408, 409, 403, 480, 232, 341]
can_val = [416, 412, 366, 454, 429, 408, 415, 186, 105, 125, 33, 85, 45, 0, 0, 0, 0, 263, 251, 90, 165]

baseline_perc = []
lego_perc = []
can_perc = []

for i in range(len(can_val)):
    if i > (len(baseline_val) - 1):
        baseline_perc.append(0)
    elif i > (len(lego_val) - 1):
        lego_perc.append(0)
    else:
        baseline_perc.append(round(baseline_val[i] / 500, 3))
        lego_perc.append(round(lego_val[i] / 500, 3))
    can_perc.append(round(can_val[i] / 500, 3))

x = np.arange(len(lables))  # the label locations
print("plt x:", x)
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, baseline_perc, width, label='Baseline')
rects2 = ax.bar(x, lego_perc, width, label='Lego Upgrade')
rects4 = ax.bar(x + width/3, can_perc, width, label='Lego Upgrade')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per class (test dataset)')
ax.set_xticks(x)
ax.set_xticklabels(lables, rotation="vertical")
ax.legend()

fig.tight_layout()

plt.show()