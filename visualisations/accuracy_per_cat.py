import os
os.chdir("visualisations")
import matplotlib.pyplot as plt
import numpy as np

lables = ["RubberToy", "PigHead", "Lego", "Can"]
acc = [0.803, 0.844, 0.84, 0.66]

plt.bar(lables, acc)

plt.show()