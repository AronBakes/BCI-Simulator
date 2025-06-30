import numpy as np
import cv2
import os

class ImagePreprocessor:
    """A class to generate and preprocess synthetic brain scan-like images.

    This class creates Gaussian blob images, preprocesses them (resize, grayscale, edge detection),
    and saves datasets for machine learning use.
    """

    def __init__(self, img_size=(128, 128), n_images=50):
        """Initialize the ImagePreprocessor with image parameters.

        Args:
            img_size (tuple, optional): Target image dimensions (height, width). Defaults to (128, 128).
            n_images (int, optional): Number of synthetic images to generate. Defaults to 50.

        Attributes:
            img_size (tuple): Target image dimensions.
            n_images (int): Number of images to generate.
        """
        self.img_size = img_size
        self.n_images = n_images

    def generate_synthetic_image(self):
        """Generate a synthetic brain scan image with Gaussian blobs.

        Returns:
            np.ndarray: Synthetic image with shape (height, width).

        Notes:
            Uses random noise and circles to mimic brain activity patterns.
        """
        img = np.zeros(self.img_size, dtype=np.uint8)
        for _ in range(np.random.randint(2, 5)):
            x, y = np.random.randint(0, self.img_size[0]), np.random.randint(0, self.img_size[1])
            sigma = np.random.uniform(10, 30)
            kernel_size = 5
            cv2.randn(img, 0, 255)
            cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma, dst=img)
            cv2.circle(img, (x, y), int(kernel_size/2), 255, -1)
        return img

    def preprocess_image(self, img):
        """Preprocess a single image with resize, grayscale, and edge detection.

        Args:
            img (np.ndarray): Input image with any shape.

        Returns:
            tuple: (img_gray, edges) where both are np.ndarray with shape (height, width).

        Notes:
            Converts to grayscale if input is BGR, applies Canny edge detection.
        """
        img_resized = cv2.resize(img, self.img_size)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
        edges = cv2.Canny(img_gray, 100, 200)
        return img_gray, edges

    def generate_dataset(self, save_path="data/raw/brain_images.npz"):
        """Generate and preprocess a dataset of synthetic brain images.

        Args:
            save_path (str, optional): Path to save the dataset as an .npz file. Defaults to "data/raw/brain_images.npz".

        Returns:
            tuple: (images, edges) where both are np.ndarray with shape (n_images, height, width).

        Notes:
            Saves images and edge maps under 'images' and 'edges' keys.
        """
        images = []
        edge_maps = []
        for _ in range(self.n_images):
            img = self.generate_synthetic_image()
            img_gray, edges = self.preprocess_image(img)
            images.append(img_gray)
            edge_maps.append(edges)
        
        images = np.array(images)
        edge_maps = np.array(edge_maps)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, images=images, edges=edge_maps)
        return images, edge_maps

    def visualize_image(self, img, edges, idx=0, save_path="data/outputs/brain_image.png"):
        """Visualize a preprocessed image and its edges.

        Args:
            img (np.ndarray): Preprocessed image with shape (height, width).
            edges (np.ndarray): Edge map with shape (height, width).
            idx (int, optional): Index of the image for labeling. Defaults to 0.
            save_path (str, optional): Path to save the plot as a PNG file. Defaults to "data/outputs/brain_image.png".

        Notes:
            Saves the plot but does not display it interactively.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image {idx} - Preprocessed")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Image {idx} - Edges")
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    preprocessor = ImagePreprocessor(img_size=(128, 128), n_images=500)
    images, edges = preprocessor.generate_dataset()
    preprocessor.visualize_image(images[0], edges[0])
    print(f"Generated {len(images)} images, shape: {images.shape}")