import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class TextPreprocessor:
    def __init__(self, n_samples=50, max_words=100):
        """
        Initialize text preprocessor for neural activity logs.
        
        Args:
            n_samples (int): Number of text samples to generate
            max_words (int): Maximum words per sample
        """
        self.n_samples = n_samples
        self.max_words = max_words
        self.vectorizer = TfidfVectorizer(max_features=500)

    def generate_synthetic_text(self):
        """
        Generate synthetic neural activity log text.
        
        Returns:
            list: List of text samples
        """
        # Sample words related to neural activity
        words = ['spike', 'wave', 'activity', 'neuron', 'signal', 'noise', 'burst', 'pattern']
        texts = []
        for _ in range(self.n_samples):
            n_words = np.random.randint(10, self.max_words)
            text = ' '.join(np.random.choice(words, n_words))
            texts.append(text)
        return texts

    def preprocess_text(self, texts):
        """
        Preprocess text data using TF-IDF vectorization.
        
        Args:
            texts (list): List of text samples
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        return tfidf_matrix

    def generate_dataset(self, save_path="data/raw/text_data.npy"):
        """
        Generate and preprocess a dataset of synthetic text.
        
        Args:
            save_path (str): Path to save the dataset
            
        Returns:
            np.ndarray: Preprocessed text features
        """
        texts = self.generate_synthetic_text()
        features = self.preprocess_text(texts)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
        return features, texts

    def print_sample(self, texts, idx=0):
        """
        Print a sample text and its features.
        
        Args:
            texts (list): List of text samples
            idx (int): Index of sample to print
        """
        print(f"Sample {idx} Text: {texts[idx]}")
        print(f"Feature shape: {self.preprocess_text([texts[idx]]).shape}")

if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(n_samples=50, max_words=100)
    features, texts = preprocessor.generate_dataset()
    preprocessor.print_sample(texts)
    print(f"Generated {len(texts)} text samples, feature shape: {features.shape}")