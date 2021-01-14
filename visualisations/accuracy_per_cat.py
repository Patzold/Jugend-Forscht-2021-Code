import os
os.chdir("visualisations")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
acc = [0.803, 0.844, 0.84, 0.66]
acc2 = [0.830, 0.484, 0.84, 0.76]

plt.bar(x-0.2, acc, 0.4)
plt.bar(x+0.2, acc2, 0.4)

plt.xticks(x, ["RubberToy", "PigHead", "Lego", "Can"])
plt.yticks(np.arange(0, 1.1, 0.1))
# plt.set_ylabel('Genauigkeit in Prozent')

plt.show()