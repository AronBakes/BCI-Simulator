import unittest
import numpy as np
from src.modeling.lstm_model import MultiModalLSTMModel

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.lstm_shape = (10, 8)  # Dummy shapes
        self.image_shape = (16384,)  # 128x128 flattened
        self.text_shape = (500,)    # TF-IDF features
        self.model = MultiModalLSTMModel(
            lstm_input_shape=self.lstm_shape,
            image_shape=self.image_shape,
            text_shape=self.text_shape
        )

    def test_model_build(self):
        self.assertIsNotNone(self.model.model)

    def test_predict_shape(self):
        dummy_lstm = np.random.rand(1, 10, 8)
        dummy_image = np.random.rand(1, 16384)
        dummy_text = np.random.rand(1, 500)
        prediction = self.model.model.predict([dummy_lstm, dummy_image, dummy_text])
        self.assertEqual(prediction.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()