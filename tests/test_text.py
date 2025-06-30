import unittest
import numpy as np
from src.preprocessing.text import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(n_samples=5, max_words=20)

    def test_generate_synthetic_text_length(self):
        texts = self.preprocessor.generate_synthetic_text()
        self.assertEqual(len(texts), 5)
        self.assertTrue(all(len(t.split()) <= 20 for t in texts))

    def test_preprocess_text_shape(self):
        texts = self.preprocessor.generate_synthetic_text()
        features = self.preprocessor.preprocess_text(texts)
        self.assertEqual(features.shape[0], 5)
        self.assertGreaterEqual(features.shape[1], 1)

if __name__ == '__main__':
    unittest.main()