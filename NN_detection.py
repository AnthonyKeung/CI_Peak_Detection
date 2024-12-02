from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as spio
import matplotlib.pyplot as plt

def peak_finding_training(windows: list, window_size: int,  peak_indices: list, model_save_path: str):

    # Creating all the data for the model
    X = np.array(windows)
    y = np.zeros((len(windows), 1))

    # Mark windows containing peaks
    for peak in peak_indices:
        for j, _ in enumerate(windows):
            if j*window_size <= peak <= (j+1)*window_size:
                y[j] = 1
                break

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        LSTM(50, input_shape=(window_size, 1)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape X for LSTM layer
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model using training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

def index_generator(windows, model_path):

    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    index = []

    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            index.append(i*window_size)

    spio.savemat('predicted_indices.mat', {'Index': index})
    return index

def class_generator(raw_data, peak_indices, model_path):

    # Creating all the data for the model
    windows = []
    for peak_index in peak_indices:
        windows.append(raw_data[peak_index:peak_index + window_size])

    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    predictions = np.argmax(predictions, axis=1).tolist()
    predictions = [prediction + 1 for prediction in predictions]
    print(predictions)
    spio.savemat('predicted_class.mat', {'Class': predictions})
    return predictions

def window_generator(raw_data, window_size):
    windows = [raw_data[i:i + window_size] for i in range(0, len(raw_data), window_size)]
    print(f"Number of windows: {len(windows)}")
    print(f"There should be {len(raw_data) // window_size} windows")
    return windows

def peak_type_training(raw_data, window_size, peak_types, peak_indices, model_save_path):

    # Creating all the data for the model
    windows = []
    for peak_index in peak_indices:
        windows.append(raw_data[peak_index:peak_index + window_size])
    
    print(f"Number of windows: {len(windows)}")

    X = np.array(windows)
    y = np.array(peak_types)
    y = y-1

    # Convert peak types to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=5)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        LSTM(50, input_shape=(window_size, 1)),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape X for LSTM layer
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model using training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

if __name__ == '__main__':

    mat = spio.loadmat('D2.mat', squeeze_me=True)
    raw_data = mat['d']
    peak_indices = mat['Index']
    peak_types = mat['Class']
    window_size = 50

    windows = window_generator(raw_data, window_size)

    # peak_finding_training(windows, window_size, peak_indices, 'peak_detection_model.h5')
    # index_generator(windows, 'peak_detection_model.h5')
    # peak_type_training(raw_data, window_size, peak_types, peak_indices, 'peak_type_model.h5')
    class_generator(raw_data, peak_indices, 'D2_peak_type_model.h5')

    