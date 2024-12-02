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
                y[j] += 1
                break
    
    # Convert y values to binary with length 2
    y = tf.keras.utils.to_categorical(y, num_classes=4)

    print(f"Number of peaks: {len(peak_indices)}")
    print(f"predicted peaks: {sum(y)}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        LSTM(50, input_shape=(window_size, 1)),
        Dense(4, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape X for LSTM layer
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model using training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

def peak_finding_training_v2(windows: list, window_size: int,  peak_indices: list, model_save_path: str):

     # Creating all the data for the model
    X = np.array(windows)
    y = np.zeros((len(windows), window_size))

    for peak in peak_indices:
        i = peak // window_size # Integer division
        j = peak % window_size # Modulus
        y[i][j] = 1


    # #Sanity check for good data in 
    # print(f"The first 10 peaks is {peak_indices[:10]}")
    # counter  = 0
    # #check the one is in the right place  
    # for i in range(len(y)):
    #     for j in range(len(y[i])):
    #         if y[i][j] == 1:
    #             counter += 1
    #             print(f"{i*window_size + j} is in the right place")
    #             if counter == 10:
    #                 return
    

   # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Define the neural network model
    model = Sequential([
        LSTM(500, input_shape=(window_size, 1)),
        Dense(200, activation='relu'),
        Dense(window_size, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model using training data
    model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

def index_generator(windows, window_size, model_path,save_path):

    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    index = []

    for i, prediction in enumerate(predictions):

        max_index = np.argmax(prediction)

        if max_index == 0: # no peak
            pass
        elif max_index == 1 : # one peak
            index.append(i*window_size)
            continue
        elif max_index == 2: # two peaks
            print(f"Two peaks: {prediction}")
            index.append(i*window_size)
            index.append(i*window_size + window_size//2)
            continue

        elif max_index == 3: # three peaks
            print(f"Three peaks: {prediction}")
            index.append(i*window_size)
            index.append(i*window_size + window_size//3)
            index.append(i*window_size - window_size//3)
            continue
        else:
            print(f"Didn't fit into any Prediction: {prediction}")
            continue

    spio.savemat(save_path, {'Index': index})
    return index
def index_generator_v2(windows, window_size, model_path,save_path): 
    
    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions[:2]}")
    index = []

    for i, prediction in enumerate(predictions):
        for j, _ in enumerate(prediction):
            if prediction[j] > 0.5:
                index.append((i*window_size) + j)

    spio.savemat(save_path, {'Index': index})
    return index

def class_generator(raw_data, peak_indices, window_size, model_path, save_path):

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
    # Load existing .mat file if it exists
    try:
        existing_data = spio.loadmat(save_path, squeeze_me=True)
        existing_data['Class'] = predictions
        spio.savemat(save_path, existing_data)
    except FileNotFoundError:
        pass
    return predictions

def window_generator(raw_data, window_size):
    windows = [raw_data[i:i + window_size] for i in range(0, len(raw_data), window_size)]
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

def generate_mat_file(peak_detection_model_path, peak_types_model_path):
    for dataset in ['D2', 'D3', 'D4', 'D5', 'D6']:
        mat = spio.loadmat(f'{dataset}.mat', squeeze_me=True)
        raw_data = mat['d']
        window_size = 50
        windows = window_generator(raw_data, window_size)

        # peak_finding_training(windows, window_size, peak_indices, 'peak_detection_model.h5')
        index_generator(windows, peak_detection_model_path,f'predicted/{dataset}.mat' )

        predicted_mat = spio.loadmat(f'predicted/{dataset}.mat', squeeze_me=True)
        peak_indices = predicted_mat['Index']

        #peak_type_training(raw_data, window_size, peak_types, peak_indices, 'peak_type_model.h5')
        class_generator(raw_data, peak_indices, window_size,  peak_types_model_path, f'predicted/{dataset}.mat')

if __name__ == '__main__':
    mat = spio.loadmat('D1.mat', squeeze_me=True)
    d = mat['d']
    Index = mat['Index']
    Index = sorted(Index)
    peak_type  = mat['Class']

    window_size = 50

    windows = window_generator(raw_data=d, window_size = window_size)
    # peak_finding_training(windows, window_size, peak_indices=Index, model_save_path='peak_finding_model.h5')
    predicted_indexes = index_generator(windows, window_size, 'peak_finding_model.h5', 'predicted.mat')

    print(f"Predicted indexes: {predicted_indexes[:10]}")
    print(len(predicted_indexes))
    print(f"Actual indexes: {Index[:10]}")
    print(len(Index))


