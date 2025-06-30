import numpy as np
import matplotlib.pyplot as plt
import os

class EEGGenerator:
    """A class to generate synthetic EEG-like time series data for brain-computer interface simulation.

    This class creates realistic EEG signals with multiple frequency components and noise,
    suitable for training machine learning models.
    """

    def __init__(self, fs=256, duration=10, n_channels=8):
        """Initialize the EEGGenerator with sampling parameters.

        Args:
            fs (int, optional): Sampling frequency in Hz. Defaults to 256.
            duration (int, optional): Duration of each signal in seconds. Defaults to 10.
            n_channels (int, optional): Number of EEG channels. Defaults to 8.

        Attributes:
            fs (int): Sampling frequency.
            duration (int): Signal duration.
            n_channels (int): Number of channels.
            t (np.ndarray): Time array for signal generation.
        """
        self.fs = fs
        self.duration = duration
        self.n_channels = n_channels
        self.t = np.linspace(0, duration, int(fs * duration))

    def generate_signal(self, freqs=[5, 10, 20], amplitudes=[1, 0.5, 0.2], noise_level=0.1):
        """Generate a synthetic EEG signal for a single channel.

        Args:
            freqs (list, optional): List of frequencies in Hz for sinusoidal components. Defaults to [5, 10, 20].
            amplitudes (list, optional): Amplitudes for each frequency component. Defaults to [1, 0.5, 0.2].
            noise_level (float, optional): Standard deviation of Gaussian noise. Defaults to 0.1.

        Returns:
            np.ndarray: Synthetic EEG signal with shape (time_points,).

        Notes:
            The signal is a sum of sine waves with added Gaussian noise.
        """
        signal = np.zeros(len(self.t))
        for f, a in zip(freqs, amplitudes):
            signal += a * np.sin(2 * np.pi * f * self.t)
        noise = np.random.normal(0, noise_level, len(self.t))
        return signal + noise

    def generate_dataset(self, n_samples=100, save_path="data/raw/eeg_data.npz"):
        """Generate a dataset of synthetic EEG signals across multiple channels and samples.

        Args:
            n_samples (int, optional): Number of EEG samples to generate. Defaults to 100.
            save_path (str, optional): Path to save the dataset as an .npz file. Defaults to "data/raw/eeg_data.npz".

        Returns:
            np.ndarray: Generated dataset with shape (n_samples, n_channels, time_points).

        Notes:
            The dataset is saved as an .npz file with a 'data' key.
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
        np.savez(save_path, data=dataset)
        return dataset

    def visualize_signal(self, signal, channel_idx=0, save_path="data/outputs/eeg_plot.png"):
        """Visualize a single EEG signal from a specified channel.

        Args:
            signal (np.ndarray): EEG signal data with shape (n_channels, time_points).
            channel_idx (int, optional): Index of the channel to visualize. Defaults to 0.
            save_path (str, optional): Path to save the plot as a PNG file. Defaults to "data/outputs/eeg_plot.png".

        Notes:
            The plot is saved but not displayed interactively.
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
    eeg_gen = EEGGenerator(fs=256, duration=10, n_channels=8)
    dataset = eeg_gen.generate_dataset(n_samples=500)
    sample = dataset[0]
    eeg_gen.visualize_signal(sample)