import os
os.chdir("visualisations")
import matplotlib.pyplot as plt
import numpy as np

lables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# names = []

one_val = [420, 446, 375, 453, 422, 406, 414, 408, 409, 403, 480, 232, 341]
two_val = [476, 492, 485, 484, 472, 475, 453, 453, 461, 475, 483, 398, 339]

one_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
two_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(one_val)):
    if one_val[i] is not 0:
        one_perc[i] = one_val[i] / 500
    if two_val[i] is not 0:
        two_perc[i] = two_val[i] / 500

print(one_perc, two_perc)

x = np.arange(len(lables))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, one_perc, width, label="FlexNet")
rects2 = ax.bar(x + width/2, two_perc, width, label="ResNet-50")
ax.set_ylabel('Genauigkeit in Prozent')
plt.yticks(np.arange(0, 1.1, 0.1))
# ax.set_title('Vergleich der Modelle')
ax.set_xticks(x)
ax.set_xticklabels(lables)
ax.legend()
ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 4, height), xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()