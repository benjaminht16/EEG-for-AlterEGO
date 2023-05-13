import os 
import numpy as np 
from scipy import signal 
import mne 
import matplotlib.pyplot as plt 
 
 
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
 
    def preprocess_eeg_signals(self, input_dir, output_dir): 
        """ 
        Preprocess EEG signals from the input directory and save the results to the output directory. 
        :param input_dir: The input directory containing raw EEG signals. 
        :param output_dir: The output directory to save the preprocessed EEG signals. 
        """ 
        # Create the output directory if it doesn't exist 
        os.makedirs(output_dir, exist_ok=True) 
 
        # Loop over the files in the input directory 
        for filename in os.listdir(input_dir): 
            # Check if the file is an EEG signal file 
            if filename.endswith('.eeg'): 
                # Preprocess the EEG signal 
                eeg_signal = self.preprocess_eeg_signal(os.path.join(input_dir, filename)) 
 
                # Save the preprocessed EEG signal to the output directory 
                with open(os.path.join(output_dir, filename), 'w') as f: 
                    f.write(eeg_signal) 
 
    def preprocess_eeg_signal(self, filepath): 
        """ 
        Preprocess a single EEG signal file. 
        :param filepath: The path to the EEG signal file. 
        :return: The preprocessed EEG signal. 
        """ 
        # Load the raw EEG signal using MNE library 
        raw = mne.io.read_raw_edf(filepath) 
        raw.load_data() 
 
        # Apply notch and line noise filters if specified 
        if self.notch_filter: 
            raw.notch_filter(np.arange(50, 251, 50), filter_length='auto', phase='zero') 
        if self.line_noise_filter: 
            raw.filter(49, 51, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, filter_length='auto') 
 
        # Extract the EEG signal 
        eeg_signal = raw.get_data() 
 
        # Remove the EOG and ECG channels from the EEG signal 
        eeg_signal = eeg_signal[:-2, :] 
 
        # Split the EEG signal into different brain waves 
        delta_wave, theta_wave, alpha_wave, low_beta_1, beta_2, high_beta_3, smr_wave, mu_wave, low_gamma_wave, high_gamma_wave = self.extract_brain_waves( 
            eeg_signal) 
 
        # Preprocess the brain waves (remove artifacts) 
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
 
        # Calculate the mean of each brain wave 
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
 
        # Create a dictionary for the means of the brain waves 
        preprocessed_eeg_signal = {"Delta Wave": delta_wave_mean, 
                                   "Theta Wave": theta_wave_mean, 
                                   "Alpha Wave": alpha_wave_mean, 
                                   "Low Beta 1 Wave": low_beta_1_mean, 
                                   "Beta 2 Wave": beta_2_mean, 
                                   "High Beta 3 Wave": high_beta_3_mean, 
                                   "SMR Wave": smr_wave_mean, 
                                   "Mu Wave": mu_wave_mean, 
                                   "Low Gamma Wave": low_gamma_wave_mean, 
                                   "High Gamma Wave": high_gamma_wave_mean} 
 
        return preprocessed_eeg_signal 
 
    def extract_brain_waves(self, eeg_signal): 
        """ 
        Split the EEG signal into different brain waves. 
        :param eeg_signal: The EEG signal to split. 
        :return: The split brain waves. 
        """ 
        # Define the frequency bands for brain waves 
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
 
        # Calculate the power spectral density of the EEG signal 
        freqs, psd = signal.welch(eeg_signal, fs=self.sfreq, nperseg=1024, average='mean') 
 
        # Extract the brain waves using the frequency bands 
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
        wave = np.mean(psd[start_idx:end_idx], axis=0) 
 
        return wave 
 
    def remove_artifacts(self, brain_wave, freq_range, max_amp): 
        """ 
        Remove artifacts from the brain wave signal. 
        :param brain_wave: The brain wave signal to remove artifacts from. 
        :param freq_range: The frequency range of the brain wave. 
        :param max_amp: The maximum amplitude of the brain wave. 
        :return: The brain wave signal with artifacts removed. 
        """ 
        # Filter the brain wave signal using the frequency range 
        filtered_wave = self.bandpass_filter(brain_wave, freq_range) 
 
        # Remove the artifacts from the brain wave signal based on the maximum amplitude 
        artifact_removed_wave = np.where(np.abs(filtered_wave) > max_amp, 0, filtered_wave) 
 
        return artifact_removed_wave 
 
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
        filtered_signal = signal.filtfilt(b, a, signal) 
 
        return filtered_signal
    def normalize_wave(self, brain_wave):  
        """  
        Normalize the brain wave signal to have zero mean and unit variance.  
        :param brain_wave: The brain wave signal to normalize.  
        :return: The normalized brain wave signal.  
        """  
        norm_wave = (brain_wave - np.mean(brain_wave)) / np.std(brain_wave)  
  
        return norm_wave  
  
    def extract_features(self, eeg_signal):  
        """  
        Extract features from the raw EEG signal.  
        :param eeg_signal: The raw EEG signal.  
        :return: The extracted features.  
        """  
        # Apply pre-processing steps to the EEG signal  
        notch_filtered_signal = self.notch_filter(eeg_signal)  
        standardized_signal = self.standardize_signal(notch_filtered_signal)  
        epoch = self.split_into_epochs(standardized_signal)  
  
        # Compute the power spectral density of the EEG signal  
        psd, freqs = self.compute_psd(epoch)  
  
        # Extract different brain waves from the power spectral density  
        delta_wave, theta_wave, alpha_wave, low_beta_1, beta_2, high_beta_3, smr_wave, mu_wave, low_gamma_wave, high_gamma_wave = self.extract_brain_waves(psd, freqs)  
  
        # Remove artifacts from the extracted brain waves  
        delta_wave = self.remove_artifacts(delta_wave, delta_band, max_amp=100)  
        theta_wave = self.remove_artifacts(theta_wave, theta_band, max_amp=75)  
        alpha_wave = self.remove_artifacts(alpha_wave, alpha_band, max_amp=50)  
        low_beta_1 = self.remove_artifacts(low_beta_1, low_beta_1_band, max_amp=30)  
        beta_2 = self.remove_artifacts(beta_2, beta_2_band, max_amp=20)  
        high_beta_3 = self.remove_artifacts(high_beta_3, high_beta_3_band, max_amp=10)  
        smr_wave = self.remove_artifacts(smr_wave, smr_band, max_amp=5)  
        mu_wave = self.remove_artifacts(mu_wave, mu_band, max_amp=3)  
        low_gamma_wave = self.remove_artifacts(low_gamma_wave, low_gamma_band, max_amp=2)  
        high_gamma_wave = self.remove_artifacts(high_gamma_wave, high_gamma_band, max_amp=1)  
  
        # Normalize the extracted brain waves  
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
  
        # Compute features from the normalized brain waves  
        feature_vector = np.concatenate((delta_wave_norm, theta_wave_norm, alpha_wave_norm, low_beta_1_norm, beta_2_norm, high_beta_3_norm, smr_wave_norm, mu_wave_norm, low_gamma_wave_norm, high_gamma_wave_norm))  
  
        return feature_vector
