from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import scipy.io as spio
import matplotlib.pyplot as plt

import NN_detection as nn


window_size = 200

mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Index = sorted(Index)
peak_type  = mat['Class']

windows = nn.window_generator(d, window_size)

# nn.peak_finding_training_v2(windows, window_size, Index, 'peak_finding_model_v2.h5')
predicted_indices = nn.index_generator_v2(windows, window_size, 'peak_finding_model_v2.h5', 'predicted_indices.mat')

print(f"The first 10 predicted indices are: {predicted_indices[:10]}")
print(f"The first 10 actual indices are: {Index[:10]}")

# nn.peak_type_training(d, window_size, peak_type, Index, 'peak_type_model.h5')
# nn.class_generator(d, Index, window_size, 'peak_type_model.h5', 'predicted_class.mat')
# mymat= spio.loadmat('predicted_class.mat', squeeze_me=True)

# predicted_peak_type = mymat['Class']

# accuracy = (predicted_peak_type == peak_type).mean() * 100
# print(f'Percentage correct: {accuracy:.2f}%')

# cm = confusion_matrix(peak_type, predicted_peak_type)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

# plt.plot(d)
# plt.plot(Index, d[Index], "x", color="red")
# plt.plot(predicted_indices, d[predicted_indices], "o", color="green")
# plt.legend(['Signal', 'Actual Peaks', 'Predicted Peaks'])

# plt.show()