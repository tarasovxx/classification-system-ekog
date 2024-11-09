import numpy as np
import scipy.signal as signal
import pywt
from scipy.fft import fft
from scipy.stats import entropy

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

# Wrapper function to extract all features
def extract_features(data, fs):
    """Extracts multiple features from the input data."""
    # Apply bandpass filtering
    filtered_data = apply_butter_bandpass_filter(data, 0.5, 30, fs)
    
    # Extract DWT features
    dwt_feats = dwt_features(filtered_data)
    
    # Extract spectral features
    spectral_feats = spectral_features(filtered_data, fs)
    
    # Calculate Hjorth parameters
    hjorth_feats = hjorth_parameters(filtered_data)
    
    # Calculate entropy features
    entropy_feats = entropy_features(filtered_data)
    
    # Concatenate all features into a single list
    all_features = dwt_feats + spectral_feats + hjorth_feats + entropy_feats
    return all_features