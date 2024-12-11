from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import scipy.io as spio
import matplotlib.pyplot as plt

import NN_detection as nn


window_size = 50

mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Indexes = mat['Index']
peak_type  = mat['Class']

for peak in range(1, 6):
    plt.figure()
    for i, Index in enumerate(Indexes):
        if peak_type[i] == peak:
            plt.plot(d[Index :Index + window_size])
    plt.title(f'Peak Type {peak}')
plt.show()

# for i, Index in enumerate(Indexes):
#     if peak_type[i] == 1:
#         plt.plot(d[Index - window_size:Index + window_size], 'r')
#     elif peak_type[i] == 2:
#         plt.plot(d[Index - window_size:Index + window_size], 'b')   
#     elif peak_type[i] == 3:
#         plt.plot(d[Index - window_size:Index + window_size], 'g')
#     elif peak_type[i] == 4:
#         plt.plot(d[Index - window_size:Index + window_size], 'y')
#     elif peak_type[i] == 5:
#         plt.plot(d[Index - window_size:Index + window_size], 'c')
#     else:
#         print('Unknown peak type')



