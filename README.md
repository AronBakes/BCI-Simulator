# BCI Simulator

A Python-based Brain-Computer Interface (BCI) simulator designed to mimic neural decoding tasks, inspired by Neuralink's mission. The project generates synthetic EEG-like time series data, trains an LSTM model to predict patterns, and processes unstructured data (brain scan images and text). Built for a Neuralink Machine Learning Engineer Intern application.

## Features
- **Synthetic EEG Data**: Generates realistic EEG-like time series using NumPy.
- **LSTM Model**: Predicts patterns with TensorFlow/Keras, targeting 70-80% accuracy.
- **Unstructured Data**: Processes brain scan images (OpenCV) and text (scikit-learn).
- **Flask Interface**: Simple web app to visualize results.
- **Testing & Optimization**: Unit tests and performance tuning for accuracy/speed.

## Installation
```bash
git clone https://github.com/yourusername/BCI_Simulator.git
cd BCI-Simulator
pip install -r requirements.txt
```

## Usage
Generate and visualize EEG data:
```bash
python src/data_generation/eeg_generator.py
```

## Project Structure
- `src/`: Core code (data generation, preprocessing, modeling, visualization, Flask app).
- `data/`: Raw, processed, and output data.
- `tests/`: Unit tests for key components.
- `configs/`: Configuration files.

## Next Steps
- Implement LSTM model for time series prediction.
- Add image and text processing pipelines.
- Deploy Flask app for result visualization.

## License
MIT License