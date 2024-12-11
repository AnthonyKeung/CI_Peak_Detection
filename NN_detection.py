import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

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

    print(f"Number of peaks: {len(peak_indices)}")
    print(f"predicted peaks: {sum(y)}")

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
        if prediction> 0.5:
            index.append(i*window_size)
    
    print(f"Number of peaks: {len(index)}")

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

def class_generator(raw_data, filtered_data, peak_indices, window_size, model_path, save_path):

    # Creating all the data for the model
    windows = []
    raw_windows = []
    for peak_index in peak_indices:
        windows.append(filtered_data[peak_index - 25 :peak_index + 25])
        raw_windows.append(raw_data[peak_index - 25 :peak_index + 25])

    
    # Ensure all windows are the same size by padding with zeros if necessary
    max_length = max(len(window) for window in windows)
    windows = [np.pad(window, (0, max_length - len(window)), 'constant') for window in windows]
    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    for i in range(10):
        i = np.random.randint(0, len(predictions))
        print(f"Prediction {i}: {max(predictions[i])}")
    predictions = np.argmax(predictions, axis=1).tolist()
    predictions = [prediction + 1 for prediction in predictions]

    print(f"For {save_path} the number of predictions are {len(predictions)}")

    for i in range(5):
        plt.subplot(5, 1, i+1)
        random_index = np.random.randint(0, len(windows))
        plt.plot(windows[random_index])
        plt.plot(raw_windows[random_index])
        plt.title(f'Window {random_index} Class {predictions[random_index]}')
    plt.tight_layout()
    plt.show()
    # Load existing .mat file if it exists
    try:
        existing_data = spio.loadmat(save_path, squeeze_me=True)
        existing_data['Class'] = predictions
        spio.savemat(save_path, existing_data)
    except FileNotFoundError:
        print(f"file not found ")
    return predictions

def window_generator(raw_data, window_size):
    windows = [raw_data[i :i + window_size ] for i in range(0, len(raw_data), window_size)]
    return windows

def class_training(raw_data, filtered_data, window_size, peak_types, peak_indices, model_save_path):

    # Making sure the peaks are at the centre of the windows
    filtered_windows = []
    for i, peak_index in enumerate(peak_indices):
        if peak_types[i] == 1:
            filtered_windows.append(filtered_data[peak_index - 18 :peak_index + 32])
        elif peak_types[i] == 2:
            filtered_windows.append(filtered_data[peak_index - 15 :peak_index + 35])
        elif peak_types[i] == 3:
            filtered_windows.append(filtered_data[peak_index - 16 :peak_index + 34])
        elif peak_types[i] == 4:
            filtered_windows.append(filtered_data[peak_index - 12 :peak_index + 38])
        elif peak_types[i] == 5:
            filtered_windows.append(filtered_data[peak_index - 11 :peak_index + 39])
    
    raw_windows = []
    for peak_index in peak_indices:
        if peak_types[i] == 1:
            raw_windows.append(raw_data[peak_index - 18 :peak_index + 32])
        elif peak_types[i] == 2:
            raw_windows.append(raw_data[peak_index - 15 :peak_index + 35])
        elif peak_types[i] == 3:
            raw_windows.append(raw_data[peak_index - 25 :peak_index + 25])
        elif peak_types[i] == 4:
            raw_windows.append(raw_data[peak_index - 12 :peak_index + 38])
        elif peak_types[i] == 5:
            raw_windows.append(raw_data[peak_index - 11 :peak_index + 39])


    for i in range(5):
        plt.subplot(5, 1, i+1)
        random_index = np.random.randint(0, len(filtered_windows))
        plt.plot(filtered_windows[random_index], 'g')
        plt.plot(raw_windows[random_index], 'r')
        plt.title(f'Window {random_index} Class {peak_types[random_index]}')
    plt.tight_layout()
    plt.show()
    
    print(f"Number of windows: {len(filtered_windows)}")

    X = np.array(filtered_windows)
    y = np.array(peak_types)
    y = y-1

    # Convert peak types to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=5)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        LSTM(60, input_shape=(window_size, 1)),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape X for LSTM layer
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model using training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_save_path)

def generate_mat_file(model_path):
    dataset   = [['D1', 50, 3700, 0.9, 0], 
                 ['D2', 70, 2300, 0.9, 0], 
                 ['D3', 150, 1250, 0.9, 0], 
                 ['D4', 150, 1200, 0.9, 0], 
                 ['D5', 230, 1050, 1.4, 0], 
                 ['D6', 210, 1050, 1.9, 0]]
    
    
    for params in dataset:
        mat = spio.loadmat(f'{params[0]}.mat', squeeze_me=True)
        raw_data = mat['d']
        window_size = 50

        low_threshold = params[1]/12500
        high_threshold = params[2]/12500
        b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
        filtered_data = filtfilt(b, a, raw_data)
        peak_indices, _ = find_peaks(x=filtered_data, height= params[3], distance=10, threshold=params[4])

        plt.figure(figsize=(10, 4))
        plt.plot(raw_data, label='Raw Data')
        plt.plot(filtered_data, label='Filtered Data')
        plt.plot(peak_indices, filtered_data[peak_indices], 'go', label='Peaks')
        plt.title(f'{params[0]} Data with Peaks')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

        print(f"For {params[0]}Number of peaks: {len(peak_indices)}")
        spio.savemat(f'predicted/{params[0]}.mat', {'Index': peak_indices})


        #peak_type_training(raw_data, window_size, peak_types, peak_indices, 'peak_type_model.h5')
        class_generator(raw_data, filtered_data, peak_indices, window_size,  model_path, f'predicted/{params[0]}.mat')

def find_thresholds(raw_data):
    number_actual_peaks = 3743
    leeway = 50

    for low in tqdm(range(200, 200, 10), desc="Low Frequency"):
        for high in range(1000, 2000, 50):
            for threshold in range(0, 5, 1):
                low_threshold = low/12500
                high_threshold = high/12500
                b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
                filtered_data = filtfilt(b, a, raw_data)
                peak_indices, _ = find_peaks(x=filtered_data, height= 0.9, distance=10, threshold=threshold/100)
                if number_actual_peaks-leeway<= len(peak_indices) <= number_actual_peaks +leeway:
                    print(f"low: {low} high: {high} number of peaks: {len(peak_indices)} threshold: {threshold}")

def view_raw_against_filtered(raw_data, low_threshold, high_threshold):
    low_threshold = low_threshold/12500
    high_threshold = high_threshold/12500
    b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
    filtered_data = filtfilt(b, a, raw_data)

    # Plot the raw data    
    plt.figure(figsize=(10, 4))
    plt.plot(raw_data)
    plt.plot(filtered_data)
    plt.title('Raw Data')
    plt.xlabel('Time')

    # Find peaks in the raw data
    peak_indices, _ = find_peaks(x=filtered_data, height= 1.9, distance=10)

    print(f"Number of peaks: {len(peak_indices)}")

    plt.plot(peak_indices, filtered_data[peak_indices], 'go')
    plt.show()



if __name__ == '__main__':
    # # Load the data from the .mat file
    # mat = spio.loadmat('D1.mat', squeeze_me=True)
    # raw_data = mat['d']
    # window_size = 50
    
    # peak_types = mat['Class']

    # # Adding Gaussian noise
    # noise = np.random.normal(0, 0.1, raw_data.shape)
    # raw_data = raw_data + noise

    # # Adding impulse noise
    # num_impulses = int(0.01 * len(raw_data))  # 1% of the data points
    # impulse_indices = np.random.randint(0, len(raw_data), num_impulses)
    # raw_data[impulse_indices] = np.random.choice([np.min(raw_data), np.max(raw_data)], num_impulses)

    # low_threshold = 50/12500
    # high_threshold = 3700/12500
    # b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
    # filtered_data = filtfilt(b, a, raw_data)
    # peak_indices = mat['Index']

    # class_training(raw_data = raw_data, filtered_data=filtered_data, window_size = window_size, 
    #                peak_types = peak_types, peak_indices=peak_indices, 
    #                model_save_path= 'filtered_peak_type_model_09_12_2024.h5')
        
    generate_mat_file(model_path='filtered_peak_type_model_09_12_2024.h5')



 
