from Perceptron import *

import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv("seeds_dataset.txt", header=None)

#Select Rosa (2) and Canadian (3)
#7th column gives species class
y = data_frame.iloc[70:210, 7].values
#if species is "2" label as -1, else (if "3") as 1
y = np.where(y == 2, -1, 1)

#Extract features (columns 0 [Area] and 4 [Length of Kernel Groove])
X = data_frame.iloc[70:210, [0, 6]].values

#Plot data
#First 70 cases in X, which we know are "Rosa"
plt.scatter(X[:70, 0], X[:70, 1], color="red", marker="o", label="Rosa")
#Second 70 cases in X, which we know are "Canadian"
plt.scatter(X[70:140, 0], X[70:140, 1], color="blue", marker="x", label="Canadian")
plt.xlabel("area of kernel [mm*2]")
plt.ylabel("length of kernel groove [mm]")
plt.legend(loc="upper left")
plt.show()