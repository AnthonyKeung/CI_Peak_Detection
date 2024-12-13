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
    """
    Trains a neural network model to detect peaks within given windows of data and saves the trained model.
    Parameters:
    windows (list): A list of data windows, where each window is a list or array of numerical values.
    window_size (int): The size of each window.
    peak_indices (list): A list of indices indicating the positions of peaks within the data.
    model_save_path (str): The file path where the trained model will be saved.
    Returns:
    None
    The function performs the following steps:
    1. Converts the list of windows into a NumPy array (X).
    2. Initializes a label array (y) with zeros.
    3. Marks windows containing peaks with a label of 1.
    4. Splits the data into training and testing sets.
    5. Defines and compiles an LSTM-based neural network model.
    6. Reshapes the input data for the LSTM layer.
    7. Trains the model using the training data.
    8. Saves the trained model to the specified file path.
    """

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

    """
    Generates indices of peaks in the given windows using a trained neural network model.
    Args:
        windows (list): A list of windows, where each window is a sequence of data points.
        window_size (int): The size of each window.
        model_path (str): The file path to the trained neural network model.
        save_path (str): The file path to save the indices of detected peaks.
    Returns:
        list: A list of indices where peaks are detected.
    """

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

def class_training(raw_data, filtered_data, window_size, peak_types, peak_indices, model_save_path,  show_graph=False):
    """
    Trains a neural network model to classify peaks in the given data.
    Parameters:
    raw_data (array-like): The raw input data.
    filtered_data (array-like): The filtered input data.
    window_size (int): The size of the window around each peak.
    peak_types (array-like): The types of peaks to classify.
    peak_indices (array-like): The indices of the peaks in the data.
    model_save_path (str): The path where the trained model will be saved.
    show_graph (bool, optional): If True, displays graphs of the windows. Default is False.
    Returns:
    None
    """

    # Making sure the peaks are at the centre of the windows
    filtered_windows = []
    for i, peak_index in enumerate(peak_indices):
        filtered_windows.append(filtered_data[peak_index :peak_index + 50])

    raw_windows = []
    for peak_index in peak_indices:
        raw_windows.append(raw_data[peak_index - 20 :peak_index + 30])

    if show_graph:
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

def class_generator(raw_data, filtered_data, peak_indices, window_size, model_path, save_path, show_graph=False):
    """
    Generates class predictions for given data using a trained neural network model.
    Parameters:
    raw_data (array-like): The raw input data.
    filtered_data (array-like): The filtered input data.
    peak_indices (list of int): Indices of peaks in the data.
    window_size (int): The size of the window to use for each peak.
    model_path (str): Path to the trained neural network model.
    save_path (str): Path to save the predictions in a .mat file.
    show_graph (bool): Whether to display a graph of the data.
    Returns:
    list: A list of predicted classes for each window.
    """

    # Creating all the data for the model
    windows = []
    raw_windows = []
    for peak_index in peak_indices:
        windows.append(filtered_data[peak_index - 20 :peak_index + 30])
        raw_windows.append(raw_data[peak_index - 20 :peak_index + 30])

    
    # Ensure all windows are the same size by padding with zeros if necessary
    max_length = max(len(window) for window in windows)
    windows = [np.pad(window, (0, max_length - len(window)), 'constant') for window in windows]
    # Convert the list of windows to a NumPy array and reshape it
    X_windows = np.array(windows).reshape(len(windows), window_size, 1)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict peaks in the test windows
    predictions = model.predict(X_windows)
    predictions = np.argmax(predictions, axis=1).tolist()
    predictions = [prediction + 1 for prediction in predictions]

    print(f"For {save_path} the number of predicted peaks are {len(predictions)}")

    if show_graph:
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

def generate_mat_file(model_path, show_graph=False):
    """
    Processes a dataset of .mat files, filters the data, detects peaks, and generates new .mat files with peak indices.
    Args:
        model_path (str): Path to the model used for class generation.
        show_graph (bool): Whether to display graphs of the data.
    Dataset format:
        Each entry in the dataset is a list with the following elements:
            - params[0] (str): Identifier for the dataset entry.
            - params[1] (float): Low threshold for the bandpass filter.
            - params[2] (float): High threshold for the bandpass filter.
            - params[3] (float): Height threshold for peak detection.
    Note:
        The function assumes that the .mat files are located in the current working directory and that the 'predicted' directory exists for saving the output files.
    """
    dataset   = [['D1', 50, 3700, 0.9], 
                 ['D2', 70, 2300, 0.9], 
                 ['D3', 150, 1250, 0.9], 
                 ['D4', 150, 1200, 0.9], 
                 ['D5', 230, 1050, 1.4], 
                 ['D6', 210, 1050, 1.9]]
    
    
    for params in dataset:
        mat = spio.loadmat(f'{params[0]}.mat', squeeze_me=True)
        raw_data = mat['d']
        window_size = 50

        low_threshold = params[1]/12500
        high_threshold = params[2]/12500
        b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
        filtered_data = filtfilt(b, a, raw_data)
        peak_indices, _ = find_peaks(x=filtered_data, height= params[3], distance=10)

        if show_graph:
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

        class_generator(raw_data, filtered_data, peak_indices, window_size,  model_path, f'predicted/{params[0]}.mat')

if __name__ == '__main__':

    ### For Training the peak finding model######################
    # Load the data from the .mat file
    mat = spio.loadmat('D1.mat', squeeze_me=True)
    raw_data = mat['d']
    window_size = 50
    
    peak_types = mat['Class']

    low_threshold = 50/12500
    high_threshold = 3700/12500
    b, a = butter(N=4, Wn=[low_threshold, high_threshold], btype='band')
    filtered_data = filtfilt(b, a, raw_data)
    peak_indices = mat['Index']

    class_training(raw_data = raw_data, filtered_data=filtered_data, window_size = window_size, 
                   peak_types = peak_types, peak_indices=peak_indices, 
                   model_save_path= 'filtered_peak_type_model_12_12_2024.keras',
                   show_graph=False)
    
     ### For Creating the mat files ######################
    generate_mat_file(model_path='filtered_peak_type_model_12_12_2024.keras', show_graph=False)



 
