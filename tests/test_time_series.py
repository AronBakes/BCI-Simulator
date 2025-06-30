import unittest
import numpy as np
from src.preprocessing.time_series import EEGPreprocessor

class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        self.preprocessor = EEGPreprocessor(window_size=256, step_size=128)
        self.data = np.random.rand(10, 8, 2560)  # Dummy EEG data

    def test_normalize_shape(self):
        normalized = self.preprocessor.normalize(self.data)
        self.assertEqual(normalized.shape, (10, 8, 2560))

    def test_normalize_stats(self):
        normalized = self.preprocessor.normalize(self.data)
        mean = np.mean(normalized, axis=(0, 2))
        std = np.std(normalized, axis=(0, 2))
        np.testing.assert_almost_equal(mean, 0, decimal=1)
        np.testing.assert_almost_equal(std, 1, decimal=1)

    def test_create_windows_shape(self):
        windows, labels = self.preprocessor.create_windows(self.data)
        self.assertGreater(windows.shape[0], 0)
        self.assertEqual(windows.shape[1:], (256, 8))

if __name__ == '__main__':
    unittest.main()