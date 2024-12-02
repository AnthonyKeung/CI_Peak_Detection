from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import scipy.io as spio
import matplotlib.pyplot as plt

mat = spio.loadmat('D1.mat', squeeze_me=True)
mymat= spio.loadmat('predicted_class.mat', squeeze_me=True)

d = mat['d']
Index = mat['Index']
peak_type  = mat['Class']

predicted_peak_type = mymat['Class']  

print(f"First 10 peaks: {peak_type[:10]}")  
print(f"First 10 predicted peaks: {predicted_peak_type[:10]}")

cm = confusion_matrix(peak_type, predicted_peak_type)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()



plt.show()
# plt.plot(d)
# plt.plot(Index, d[Index], "x", color="red")
# plt.plot(predicted_indices, d[predicted_indices], "o", color="green")
# plt.legend(['Signal', 'Actual Peaks', 'Predicted Peaks'])

# plt.show()