import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import threading
import asyncio
import buffer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy import signal
from adaptive_filter import AdaptiveFilter
from deep_learning_extractor import DeepLearningExtractor
from alter_echo_interface import AlterEchoInterface
from buffer import Buffer

class EEGProcessor:
    def __init__(self, input_dir, output_dir, notch_filter=True, line_noise_filter=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.notch_filter = notch_filter
        self.line_noise_filter = line_noise_filter
        self.sfreq = 256
        self.buffer = Buffer(max_size=10)  # create a buffer with a maximum size of 10

    def process_eeg_signals(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.eeg'):
                eeg_signal = self.preprocess_eeg_signal(os.path.join(self.input_dir, filename))
                self.buffer.write(eeg_signal)  # write the preprocessed EEG signal to the buffer

    def process_features(self):
        X, y = self.collect_features_and_labels()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        clf = self.train_classifier(X_train, y_train)
        accuracy = self.evaluate_classifier(clf, X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}")

    def collect_features_and_labels(self):
        X = []
        y = []
        while True:
            eeg_signal = self.buffer.get_latest_data()
            if eeg_signal is None:
                break
            features = self.extract_features(eeg_signal)
            X.append(features)
            y.append(self.get_label_from_filename(filename))  # get the label for the EEG signal from the filename
        X = np.array(X)
        y = np.array(y)
        return X, y

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_classifier(self, X_train, y_train):
        scaler = StandardScaler()
        clf = make_pipeline(scaler, RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        clf.fit(X_train, y_train)
        return clf

    def evaluate_classifier(self, clf, X_test, y_test):
        return clf.score(X_test, y_test)

class my_class:
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    low_beta_1_band = (12.5, 16.0)
    beta_2_band = (16.5, 20.0)
    high_beta_3_band = (20.5, 28.0)
    smr_band = (12.0, 15.0)
    mu_band = (8.0, 13.0)
    low_gamma_band = (30.0, 50.0)
    high_gamma_band = (50.0, 100.0)

class EEGWaveExtractor:
    """ 
    Preprocess EEG signals and extract different brain waves. 
    """ 
 
    def __init__(self, sfreq=256, notch_filter=True, line_noise_filter=True):
        """
        Initialize the EEGWaveExtractor object.
        :param sfreq: The sampling frequency of the EEG signal.
        :param notch_filter: Whether to apply a notch filter to remove power line noise.
        :param line_noise_filter: Whether to apply a line noise filter to remove line frequency noise.
        """
        self.sfreq = sfreq
        self.notch_filter = notch_filter
        self.line_noise_filter = line_noise_filter
        self.buffer = buffer.Buffer()
        self.alter_echo_interface = AlterEchoInterface()
        self.adaptive_filter = AdaptiveFilter()
        self.deep_learning_extractor = DeepLearningExtractor()
        self.processing_thread = threading.Thread(target=self.realtime_processing)
        self.processing_thread.start()

    def realtime_processing(self):
        """
        Process EEG signals in real-time.
        """
        while True:
            eeg_signal = self.buffer.get_latest_data()
            if len(eeg_signal) > 0:
                eeg_signal = self.adaptive_filter.apply(eeg_signal)
                feature_vector = self.extract_features(eeg_signal)
                self.alter_echo_interface.send_features(feature_vector)
                time.sleep(0.1)

    def preprocess_eeg_signals(self, input_dir, output_dir): 
        """ 
        Preprocess EEG signals from the input directory and save the results to the output directory. 
        :param input_dir: The input directory containing raw EEG signals. 
        :param output_dir: The output directory to save the preprocessed EEG signals. 
        """  
        os.makedirs(output_dir, exist_ok=True) 
 
        for filename in os.listdir(input_dir):  
            if filename.endswith('.eeg'): 
                eeg_signal = self.preprocess_eeg_signal(os.path.join(input_dir, filename)) 
 
                with open(os.path.join(output_dir, filename), 'w') as f: 
                    f.write(eeg_signal) 
 
    def preprocess_eeg_signal(self, filepath):
        """ 
        Preprocess a single EEG signal file. 
        :param filepath: The path to the EEG signal file. 
        :return: The preprocessed EEG signal. 
        """  
        raw = mne.io.read_raw_edf(filepath)
        raw.load_data() 

        if self.notch_filter: 
            raw.notch_filter(np.arange(50, 251, 50), filter_length='auto', phase='zero')
        if self.line_noise_filter: 
            raw.filter(49, 51, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, filter_length='auto') 

        eeg_signal = raw.get_data() 

        eeg_signal = eeg_signal[:-2, :] 

        delta_wave, theta_wave, alpha_wave, low_beta_1, beta_2, high_beta_3, smr_wave, mu_wave, low_gamma_wave, high_gamma_wave = self.extract_brain_waves( 
            eeg_signal) 

        delta_wave = self.remove_artifacts(delta_wave, freq_range=(0.5, 4.0), max_amp=200e-6)
        theta_wave = self.remove_artifacts(theta_wave, freq_range=(4.0, 8.0), max_amp=100e-6)
        alpha_wave = self.remove_artifacts(alpha_wave, freq_range=(8.0, 13.0), max_amp=50e-6)
        low_beta_1 = self.remove_artifacts(low_beta_1, freq_range=(12.5, 16.0), max_amp=30e-6)
        beta_2 = self.remove_artifacts(beta_2, freq_range=(16.5, 20.0), max_amp=20e-6)
        high_beta_3 = self.remove_artifacts(high_beta_3, freq_range=(20.5, 28.0), max_amp=15e-6)
        smr_wave = self.remove_artifacts(smr_wave, freq_range=(12.0, 15.0), max_amp=30e-6)
        mu_wave = self.remove_artifacts(mu_wave, freq_range=(8.0, 13.0), max_amp=50e-6)
        low_gamma_wave = self.remove_artifacts(low_gamma_wave, freq_range=(30.0, 50.0), max_amp=10e-6)
        high_gamma_wave = self.remove_artifacts(high_gamma_wave, freq_range=(50.0, 100.0), max_amp=5e-6) 

        delta_wave_mean = np.mean(delta_wave)
        theta_wave_mean = np.mean(theta_wave)
        alpha_wave_mean = np.mean(alpha_wave)
        low_beta_1_mean = np.mean(low_beta_1)
        beta_2_mean = np.mean(beta_2)
        high_beta_3_mean = np.mean(high_beta_3)
        smr_wave_mean = np.mean(smr_wave)
        mu_wave_mean = np.mean(mu_wave)
        low_gamma_wave_mean = np.mean(low_gamma_wave)
        high_gamma_wave_mean = np.mean(high_gamma_wave) 

        return {
            "Delta Wave": delta_wave_mean,
            "Theta Wave": theta_wave_mean,
            "Alpha Wave": alpha_wave_mean,
            "Low Beta 1 Wave": low_beta_1_mean,
            "Beta 2 Wave": beta_2_mean,
            "High Beta 3 Wave": high_beta_3_mean,
            "SMR Wave": smr_wave_mean,
            "Mu Wave": mu_wave_mean,
            "Low Gamma Wave": low_gamma_wave_mean,
            "High Gamma Wave": high_gamma_wave_mean,
        } 
 
    def extract_brain_waves(self, eeg_signal): 
        """ 
        Split the EEG signal into different brain waves. 
        :param eeg_signal: The EEG signal to split. 
        :return: The split brain waves. 
        """  
        delta_band = (0.5, 4.0) 
        theta_band = (4.0, 8.0) 
        alpha_band = (8.0, 13.0) 
        low_beta_1_band = (12.5, 16.0) 
        beta_2_band = (16.5, 20.0) 
        high_beta_3_band = (20.5, 28.0) 
        smr_band = (12.0, 15.0) 
        mu_band = (8.0, 13.0) 
        low_gamma_band = (30.0, 50.0) 
        high_gamma_band = (50.0, 100.0) 
  
        freqs, psd = signal.welch(eeg_signal, fs=self.sfreq, nperseg=1024, average='mean') 
  
        delta_wave = self.extract_wave(psd, freqs, delta_band) 
        theta_wave = self.extract_wave(psd, freqs, theta_band) 
        alpha_wave = self.extract_wave(psd, freqs, alpha_band) 
        low_beta_1 = self.extract_wave(psd, freqs, low_beta_1_band) 
        beta_2 = self.extract_wave(psd, freqs, beta_2_band) 
        high_beta_3 = self.extract_wave(psd, freqs, high_beta_3_band) 
        smr_wave = self.extract_wave(psd, freqs, smr_band) 
        mu_wave = self.extract_wave(psd, freqs, mu_band) 
        low_gamma_wave = self.extract_wave(psd, freqs, low_gamma_band) 
        high_gamma_wave = self.extract_wave(psd, freqs, high_gamma_band) 
 
        return delta_wave, theta_wave, alpha_wave, low_beta_1, beta_2, high_beta_3, smr_wave, mu_wave, low_gamma_wave, high_gamma_wave 
 
    @staticmethod
    def extract_wave(psd, freqs, freq_range):
        """ 
        Extract a specific brain wave from the power spectral density of the EEG signal within the given frequency range. 
        :param psd: The power spectral density of the EEG signal. 
        :param freqs: The frequencies corresponding to the power spectral density. 
        :param freq_range: The frequency range to extract the brain wave. 
        :return: The extracted brain wave. 
        """ 
        start_idx, end_idx = np.searchsorted(freqs, freq_range)
        return np.mean(psd[start_idx:end_idx], axis=0) 
 
    def remove_artifacts(self, brain_wave, freq_range, max_amp):
        """ 
        Remove artifacts from the brain wave signal. 
        :param brain_wave: The brain wave signal to remove artifacts from. 
        :param freq_range: The frequency range of the brain wave. 
        :param max_amp: The maximum amplitude of the brain wave. 
        :return: The brain wave signal with artifacts removed. 
        """ 
        filtered_wave = self.bandpass_filter(brain_wave, freq_range) 

        return np.where(np.abs(filtered_wave) > max_amp, 0, filtered_wave) 
 
    def bandpass_filter(self, signal, freq_range):
        """ 
        Apply a bandpass filter to the signal using the given frequency range. 
        :param signal: The signal to apply the bandpass filter to. 
        :param freq_range: The frequency range to apply the bandpass filter. 
        :return: The filtered signal. 
        """ 
        nyquist_freq = self.sfreq / 2
        low_freq, high_freq = freq_range
        low_freq /= nyquist_freq
        high_freq /= nyquist_freq
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        return signal.filtfilt(b, a, signal)
    def normalize_wave(self, brain_wave):
        """  
        Normalize the brain wave signal to have zero mean and unit variance.  
        :param brain_wave: The brain wave signal to normalize.  
        :return: The normalized brain wave signal.  
        """  
        return (brain_wave - np.mean(brain_wave)) / np.std(brain_wave)  

    def extract_features(self, eeg_signal):
        """
        Extract features from the raw EEG signal.
        :param eeg_signal: The raw EEG signal.
        :return: The extracted features.
        """

        notch_filtered_signal = self.notch_filter(eeg_signal)
        standardized_signal = self.standardize_signal(notch_filtered_signal)
        epoch = self.split_into_epochs(standardized_signal)  

        psd, freqs = self.compute_psd(epoch)  

        delta_wave, theta_wave, alpha_wave, low_beta_1, beta_2, high_beta_3, smr_wave, mu_wave, low_gamma_wave, high_gamma_wave = self.extract_brain_waves(psd, freqs)

        delta_wave = self.remove_artifacts(delta_wave, my_class.delta_band, max_amp=100)
        theta_wave = self.remove_artifacts(theta_wave, my_class.theta_band, max_amp=75)
        alpha_wave = self.remove_artifacts(alpha_wave, my_class.alpha_band, max_amp=50)
        low_beta_1 = self.remove_artifacts(low_beta_1, my_class.low_beta_1_band, max_amp=30)
        beta_2 = self.remove_artifacts(beta_2, my_class.beta_2_band, max_amp=20)
        high_beta_3 = self.remove_artifacts(high_beta_3, my_class.high_beta_3_band, max_amp=10)
        smr_wave = self.remove_artifacts(smr_wave, my_class.smr_band, max_amp=5)
        mu_wave = self.remove_artifacts(mu_wave, my_class.mu_band, max_amp=3)
        low_gamma_wave = self.remove_artifacts(low_gamma_wave, my_class.low_gamma_band, max_amp=2)
        high_gamma_wave = self.remove_artifacts(high_gamma_wave, my_class.high_gamma_band, max_amp=1)  

        delta_wave_norm = self.normalize_wave(delta_wave)
        theta_wave_norm = self.normalize_wave(theta_wave)
        alpha_wave_norm = self.normalize_wave(alpha_wave)
        low_beta_1_norm = self.normalize_wave(low_beta_1)
        beta_2_norm = self.normalize_wave(beta_2)
        high_beta_3_norm = self.normalize_wave(high_beta_3)
        smr_wave_norm = self.normalize_wave(smr_wave)
        mu_wave_norm = self.normalize_wave(mu_wave)
        low_gamma_wave_norm = self.normalize_wave(low_gamma_wave)
        high_gamma_wave_norm = self.normalize_wave(high_gamma_wave)  

        return self.deep_learning_extractor.compute_features(
            delta_wave_norm,
            theta_wave_norm,
            alpha_wave_norm,
            low_beta_1_norm,
            beta_2_norm,
            high_beta_3_norm,
            smr_wave_norm,
            mu_wave_norm,
            low_gamma_wave_norm,
            high_gamma_wave_norm,
        )
