import os
os.chdir("visualisations")
import matplotlib.pyplot as plt
import numpy as np

lables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# names = []

baseline_val = [443, 449, 407, 481, 434, 473, 477, 0, 0, 0, 0, 0, 0]
lego_val = [420, 446, 375, 453, 422, 406, 414, 408, 409, 403, 480, 232, 341]
can_val = [416, 412, 366, 454, 429, 408, 415, 186, 105, 125, 33, 85, 45, 0, 0, 0, 0, 263, 251, 90, 165]

baseline_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lego_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
can_perc = []

for i in range(len(lego_val)):
    if baseline_val[i] is not 0:
        baseline_perc[i] = baseline_val[i] / 10
    if lego_val[i] is not 0:
        lego_perc[i] = lego_val[i] / 10

print(baseline_val, lego_val)

x = np.arange(len(lables))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, baseline_perc, width) # label='Grundlage'
rects2 = ax.bar(x + width/2, lego_perc, width) #  label='+ Lego'

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Genauigkeit in Prozent')
ax.set_title('Genauigkeit nach Klasse')
ax.set_xticks(x)
ax.set_xticklabels(lables)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 4, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()