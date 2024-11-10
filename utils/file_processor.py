import pandas as pd
import re
import glob 
import os

import numpy as np
import scipy.signal as signal
import pywt
from scipy.fft import fft
from scipy.stats import entropy
import pyedflib
import warnings

warnings.filterwarnings("ignore")

# Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter with the given cutoff frequencies."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# Discrete Wavelet Transform (DWT) for feature extraction
def dwt_features(data, wavelet='db4', level=5):
    """Applies Discrete Wavelet Transform and extracts features from the coefficients."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Extract energy from each level of DWT coefficients
    features = [np.sum(np.square(c)) for c in coeffs]
    return features

# Spectral features (e.g., power spectral density)
def spectral_features(data, fs):
    """Calculates spectral features such as spectral power."""
    freqs, psd = signal.welch(data, fs)
    # Example: Extracting power in different frequency bands
    delta_power = np.trapz(psd[(freqs >= 0.5) & (freqs <= 4)])
    theta_power = np.trapz(psd[(freqs >= 4) & (freqs <= 8)])
    alpha_power = np.trapz(psd[(freqs >= 8) & (freqs <= 13)])
    beta_power = np.trapz(psd[(freqs >= 13) & (freqs <= 30)])
    return [delta_power, theta_power, alpha_power, beta_power]

# Hjorth Parameters
def hjorth_parameters(data):
    """Calculates Hjorth parameters: activity, mobility, and complexity."""
    activity = np.var(data)
    mobility = np.sqrt(np.var(np.diff(data)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(data))) / np.var(np.diff(data)))
    return [activity, mobility, complexity]

# Entropy measures
def entropy_features(data):
    """Calculates entropy measures (e.g., Shannon entropy)."""
    hist, _ = np.histogram(data, bins=256, density=True)
    shannon_entropy = entropy(hist, base=2)  # Shannon Entropy
    return [shannon_entropy]

def extract_features(data, fs):
    """Extracts multiple features from the input data and returns feature names along with values."""
    # Apply bandpass filtering
    filtered_data = apply_butter_bandpass_filter(data, 0.5, 30, fs)
    
    # Extract DWT features
    dwt_feats = dwt_features(filtered_data)
    dwt_feat_names = [f'DWT_Energy_Level_{i+1}' for i in range(len(dwt_feats))]

    # Extract spectral features
    spectral_feats = spectral_features(filtered_data, fs)
    spectral_feat_names = ['Delta_Power', 'Theta_Power', 'Alpha_Power', 'Beta_Power']

    # Calculate Hjorth parameters
    hjorth_feats = hjorth_parameters(filtered_data)
    hjorth_feat_names = ['Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity']

    # Calculate entropy features
    entropy_feats = entropy_features(filtered_data)
    entropy_feat_names = ['Shannon_Entropy']

    # Concatenate all features and names into single lists
    all_features = dwt_feats + spectral_feats + hjorth_feats + entropy_feats
    all_feature_names = dwt_feat_names + spectral_feat_names + hjorth_feat_names + entropy_feat_names
    
    return all_features, all_feature_names



# Function to convert time strings to seconds
def time_to_seconds(t):
    try:
        if pd.isnull(t) or t == '':
            return None
        parts = list(map(int, re.split('[:]', str(t))))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 1:
            return parts[0]
        else:
            return None
    except ValueError:
        return None

def parse_intervals(file_path):
    # Extract metadata from filename
    filename_id = file_path.split('/')[-1].split('_')

    id_match = filename_id[0]
    age_match = filename_id[1]
    exp_match = filename_id[2]
    
    # Assign extracted values or None if not found
    
    # Read the file into a DataFrame
    intervals_df = pd.read_csv(file_path, sep='\t', header=None, names=['NN', 'Время', 'Маркер'])
    intervals_df['Начало'] = intervals_df['Время'].apply(time_to_seconds)

    # Ensure 'Начало' column is numeric, removing non-numeric values
    intervals_df['Начало'] = pd.to_numeric(intervals_df['Начало'], errors='coerce')
    intervals_df = intervals_df.dropna(subset=['Начало']).reset_index(drop=True)

    paired_intervals = []
    open_markers_last_time = {}

    for _, row in intervals_df.iterrows():
        marker_base = re.match(r"([a-zA-Z]+)", row['Маркер']).group(0)
        
        if marker_base in open_markers_last_time:
            last_time = open_markers_last_time[marker_base]
            if isinstance(row['Начало'], (int, float)) and isinstance(last_time, (int, float)):
                if row['Начало'] > last_time:
                    paired_intervals.append((last_time, row['Начало'], marker_base))

        open_markers_last_time[marker_base] = row['Начало']

    # Create a DataFrame with valid pairs
    if paired_intervals:
        valid_intervals_df = pd.DataFrame(
            paired_intervals,
            columns=['X_FROM', 'X_TO', 'LABEL']
        )
        # Add extracted metadata as new columns
        valid_intervals_df['ID'] = id_match
        valid_intervals_df['AGE_MONTHS'] = age_match
        valid_intervals_df['EXPERIMENT_TYPE'] = exp_match
    else:
        valid_intervals_df = pd.DataFrame(columns=['X_FROM', 'X_TO', 'LABEL', 'ID', 'AGE_MONTHS', 'EXPERIMENT_TYPE'])

    return valid_intervals_df

def process_file(file_path):
    valid_intervals = parse_intervals(file_path)
        
    with pyedflib.EdfReader(file_path[:-4]+".edf") as edf_reader:
        print(file_path[:-4]+".edf")
        signal_labels = edf_reader.getSignalLabels()
        # Specify the desired channel
        feature_rows = []

        for types in signal_labels:
            channel_index = signal_labels.index(types)
            data = edf_reader.readSignal(channel_index)
            sfreq = edf_reader.getSampleFrequency(channel_index)
            total_duration = len(data) / sfreq

            for _, row in valid_intervals.iterrows():
                start_time = row['X_FROM']
                duration = row['X_TO'] - row['X_FROM']
                end_time = start_time + duration

                if end_time > total_duration:
                    end_time = total_duration
                    duration = end_time - start_time

                # Convert times to indices
                start_idx = int(start_time * sfreq)
                end_idx = int(end_time * sfreq)
                # Extract the signal segment
                signal_data = data[start_idx:end_idx]
                # print(signal_data)
                # Extract features from this segment
                features, feature_names = extract_features(signal_data, 256)
                # Convert features to DataFrame row and transpose
                feature_row = pd.DataFrame([features]).T
                # print(feature_row)

                feature_row = feature_row.transpose()
                feature_row.columns = [f"{i}_{types}" for i in feature_names]

                # Append the feature row
                feature_rows.append(feature_row)

            # Concatenate all feature rows with the original DataFrame
        features_df = pd.concat(feature_rows, ignore_index=True)
        return  pd.concat([valid_intervals.reset_index(drop=True), features_df], axis=1)


# Function to process all .txt files in a folder
def process_folder(folder_path):
    all_intervals = []
    
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            df_with_features = process_file(file_path)
            all_intervals.append(df_with_features)

    # Concatenate all DataFrames
    all_intervals_df = pd.concat(all_intervals, ignore_index=True)
    return all_intervals_df


