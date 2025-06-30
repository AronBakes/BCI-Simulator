from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load sample data (replace with actual data loading logic)
    eeg_data = np.load("data/processed/eeg_processed.npz")['windows'][0]
    image_data = np.load("data/raw/brain_images.npz")['images'][0]

    # Create plots
    plt.figure(figsize=(10, 4))
    plt.plot(eeg_data.T)  # Transpose for channels vs. time
    plt.title("Sample EEG Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    eeg_plot = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(image_data, cmap='gray')
    plt.title("Sample Brain Image")
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    image_plot = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render_template('index.html', eeg_plot=eeg_plot, image_plot=image_plot)

if __name__ == '__main__':
    app.run(debug=True)