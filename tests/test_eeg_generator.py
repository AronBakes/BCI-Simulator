import unittest
import numpy as np
from src.data_generation.eeg_generator import EEGGenerator

class TestEEGGenerator(unittest.TestCase):
    def setUp(self):
        self.eeg_gen = EEGGenerator(fs=256, duration=10, n_channels=8)

    def test_generate_signal_shape(self):
        signal = self.eeg_gen.generate_signal()
        self.assertEqual(len(signal), 2560)  # 256 Hz * 10 s

    def test_generate_dataset_shape(self):
        dataset = self.eeg_gen.generate_dataset(n_samples=5)
        self.assertEqual(dataset.shape, (5, 8, 2560))

    def test_signal_statistics(self):
        signal = self.eeg_gen.generate_signal()
        mean = np.mean(signal)
        std = np.std(signal)
        self.assertTrue(-0.1 < mean < 0.1)  # Mean should be near zero
        self.assertTrue(0.5 < std < 1.5)    # Std should be reasonable with noise

if __name__ == '__main__':
    unittest.main()