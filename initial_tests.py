from scipy.signal import find_peaks

import scipy.io as spio
import matplotlib.pyplot as plt

mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Index = Index[0:10]

print(f"The index of the peaks are {Index}")
print(f"The value of the peaks are {d[Index]}")

plt.plot(d)
plt.plot(Index, d[Index], "x")
plt.show()