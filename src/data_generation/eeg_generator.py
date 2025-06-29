import numpy as np
import matplotlib.pyplot as plt
import os

class EEGGenerator:
    def __init__(self, fs=256, duration=10, n_channels=8):
        """
        Initialize EEG-like data generator.
        
        Args:
            fs (int): Sampling frequency (Hz)
            duration (int): Duration of signal (seconds)
            n_channels (int): Number of EEG channels
        """
        self.fs = fs
        self.duration = duration
        self.n_channels = n_channels
        self.t = np.linspace(0, duration, int(fs * duration))

    def generate_signal(self, freqs=[5, 10, 20], amplitudes=[1, 0.5, 0.2], noise_level=0.1):
        """
        Generate synthetic EEG signal for one channel.
        
        Args:
            freqs (list): Frequencies for sinusoidal components (Hz)
            amplitudes (list): Amplitudes for each frequency
            noise_level (float): Standard deviation of Gaussian noise
            
        Returns:
            np.ndarray: Synthetic EEG signal
        """
        signal = np.zeros(len(self.t))
        for f, a in zip(freqs, amplitudes):
            signal += a * np.sin(2 * np.pi * f * self.t)
        noise = np.random.normal(0, noise_level, len(self.t))
        return signal + noise

    def generate_dataset(self, n_samples=100, save_path="data/raw/eeg_data.npy"):
        """
        Generate dataset with multiple EEG samples.
        
        Args:
            n_samples (int): Number of samples to generate
            save_path (str): Path to save the dataset
            
        Returns:
            np.ndarray: EEG dataset (n_samples, n_channels, time_points)
        """
        dataset = []
        for _ in range(n_samples):
            sample = []
            for _ in range(self.n_channels):
                signal = self.generate_signal()
                sample.append(signal)
            dataset.append(sample)
        dataset = np.array(dataset)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, dataset)
        return dataset

    def visualize_signal(self, signal, channel_idx=0, save_path="data/outputs/eeg_plot.png"):
        """
        Visualize a single EEG signal.
        
        Args:
            signal (np.ndarray): EEG signal to plot
            channel_idx (int): Channel index to visualize
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, signal[channel_idx], label=f"Channel {channel_idx+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Synthetic EEG Signal")
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Example usage
    eeg_gen = EEGGenerator(fs=256, duration=10, n_channels=8)
    dataset = eeg_gen.generate_dataset(n_samples=100)
    sample = dataset[0]  # First sample
    eeg_gen.visualize_signal(sample)
