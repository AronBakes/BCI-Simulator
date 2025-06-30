import unittest
import numpy as np
from src.preprocessing.image import ImagePreprocessor

class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor(img_size=(128, 128), n_images=5)

    def test_generate_synthetic_image_shape(self):
        img = self.preprocessor.generate_synthetic_image()
        self.assertEqual(img.shape, (128, 128))

    def test_preprocess_image_shape(self):
        img = self.preprocessor.generate_synthetic_image()
        img_gray, edges = self.preprocessor.preprocess_image(img)
        self.assertEqual(img_gray.shape, (128, 128))
        self.assertEqual(edges.shape, (128, 128))

    def test_generate_dataset_shape(self):
        images, edges = self.preprocessor.generate_dataset(save_path="data/raw/test_images.npz")
        self.assertEqual(images.shape, (5, 128, 128))

if __name__ == '__main__':
    unittest.main()