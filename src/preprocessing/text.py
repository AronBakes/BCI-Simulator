import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class TextPreprocessor:
    """A class to generate and preprocess synthetic neural activity log text.

    This class creates random text samples and converts them into TF-IDF features
    for machine learning applications.
    """

    def __init__(self, n_samples=50, max_words=100):
        """Initialize the TextPreprocessor with text generation parameters.

        Args:
            n_samples (int, optional): Number of text samples to generate. Defaults to 50.
            max_words (int, optional): Maximum words per sample. Defaults to 100.

        Attributes:
            n_samples (int): Number of samples.
            max_words (int): Maximum words per sample.
            vectorizer (TfidfVectorizer): TF-IDF vectorizer instance.
        """
        self.n_samples = n_samples
        self.max_words = max_words
        self.vectorizer = TfidfVectorizer(max_features=500)

    def generate_synthetic_text(self):
        """Generate synthetic neural activity log text.

        Returns:
            list: List of text samples, each a string of random neural-related words.

        Notes:
            Uses a fixed set of words to simulate activity logs.
        """
        words = ['spike', 'wave', 'activity', 'neuron', 'signal', 'noise', 'burst', 'pattern']
        texts = []
        for _ in range(self.n_samples):
            n_words = np.random.randint(10, self.max_words)
            text = ' '.join(np.random.choice(words, n_words))
            texts.append(text)
        return texts

    def preprocess_text(self, texts):
        """Preprocess text data using TF-IDF vectorization.

        Args:
            texts (list): List of text samples.

        Returns:
            np.ndarray: TF-IDF feature matrix with shape (n_samples, n_features).

        Notes:
            Limits to 500 features for efficiency.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        return tfidf_matrix

    def generate_dataset(self, save_path="data/raw/text_data.npz"):
        """Generate and preprocess a dataset of synthetic text.

        Args:
            save_path (str, optional): Path to save the dataset as an .npz file. Defaults to "data/raw/text_data.npz".

        Returns:
            tuple: (features, texts) where features is np.ndarray and texts is list.

        Notes:
            Saves texts and features under 'texts' and 'features' keys.
            Includes error handling for file operations.
        """
        texts = self.generate_synthetic_text()
        features = self.preprocess_text(texts)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            np.savez(save_path, texts=texts, features=features)
            print(f"Successfully saved to {save_path}")
        except Exception as e:
            print(f"Error saving to {save_path}: {e}")
            raise
        return features, texts

    def print_sample(self, texts, idx=0):
        """Print a sample text and its TF-IDF features.

        Args:
            texts (list): List of text samples.
            idx (int, optional): Index of the sample to print. Defaults to 0.

        Notes:
            Features are recomputed for the single sample.
        """
        print(f"Sample {idx} Text: {texts[idx]}")
        print(f"Feature shape: {self.preprocess_text([texts[idx]]).shape}")

if __name__ == "__main__":
    preprocessor = TextPreprocessor(n_samples=500, max_words=100)
    features, texts = preprocessor.generate_dataset()
    preprocessor.print_sample(texts)