from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from sklearn.model_selection import train_test_split


import scipy.io as spio
import matplotlib.pyplot as plt

# mat = spio.loadmat('D1.mat', squeeze_me=True)
# raw_data = mat['d']

# peak_indices = mat['Index']
# peak_indices = sorted(peak_indices)

# #split raw data into window size of 100 
# window_size = 100
# windows = [raw_data[i:i + window_size] for i in range(0, len(raw_data), window_size)]

# print(f"Number of windows: {len(windows)}")
# print(f"There should be {len(raw_data) // window_size} windows")

# # Creating all the data for the model
# X = np.array(windows)
# y = np.zeros((len(windows), 1))

# # Mark windows containing peaks
# for peak in peak_indices:
#     for j, window in enumerate(windows):
#         if j*window_size <= peak <= (j+1)*window_size:
#             y[j] = 1
#             print(f"{peak} is inbetween {j*window_size} and {(j+1)*window_size}")
#             break


            

# # Print the number of 1s in disparity dues to multiple peaks in windows
# print(f"Number of windows containing peaks: {np.sum(y)}")
# print(f"Number of actual peaks: {len(peak_indices)}")

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # # Define the neural network model
# # model = Sequential([
# #     LSTM(50, input_shape=(window_size, 1)),
# #     Dense(1, activation='sigmoid')
# # ])

# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # # Reshape X for LSTM layer
# # X = X.reshape((X.shape[0], X.shape[1], 1))

# # # Train the model using training data
# # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# # # Save the trained model
# # model.save('peak_detection_model.h5')


mat = spio.loadmat('D1.mat', squeeze_me=True)
raw_data = mat['d'][:20000]
actual_peak_indices = sorted(mat['Index'])
actual_peak_indices = actual_peak_indices[:10]
raw_data = raw_data

window_size = 50
windows = [raw_data[i:i + window_size] for i in range(0, len(raw_data), window_size)]
peak_count = 0 

# Predict peaks in the test windows
model = tf.keras.models.load_model('peak_detection_model.h5')
peak_count = 0

for i, window in enumerate(windows):
    prediction = model.predict(window.reshape(1, window_size, 1))
    if prediction > 0.5:
        plt.scatter(range(i*window_size, (i+1)*window_size), window, color='red', marker='x')
        peak_count += 1

plt.plot(raw_data, label='Raw Data')
plt.plot(actual_peak_indices, raw_data[actual_peak_indices], 'x', label='Actual Peaks', color='green')

plt.title('Raw Data with Peaks')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
print(f"Total number of peaks detected: {peak_count}")

