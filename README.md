# Python speech recognition and audio classification automation

This Python automation project extracts acoustic features from speech recordings and trains neural network classifiers to transcribe audio.

## Purpose

The project provides an automated pipeline in Python for speech recognition, LibriSpeech dataset processing, audio feature extraction, and recurrent neural network training.

## Architecture

The system processes audio through five sequential stages:

1. Data ingestion: The loader reads FLAC audio files and plain text transcription files from the LibriSpeech corpus directory.
2. Audio preprocessing: The pipeline resamples audio to 16000 Hz if the source sample rate differs. It extracts 13 Mel-Frequency Cepstral Coefficients per frame with Librosa, then averages the coefficients across time to generate a fixed 13-element feature vector for each recording.
3. Label encoding: Scikit-learn transforms text transcriptions into integer class indices.
4. Model training: A TensorFlow Keras sequential model receives the 13 inputs. The model contains a 128-unit dense layer with rectified linear activation, a 20 percent dropout rate, a 64-unit dense layer, a bidirectional LSTM layer with 128 units, a second 20 percent dropout rate, a forward LSTM layer with 128 units, and a final dense layer with softmax activation for classification. The optimizer is Adam and the loss function is sparse categorical cross-entropy.
5. Evaluation and plotting: The script evaluates classification accuracy on an 80 percent training and 20 percent testing split, prints test loss and test accuracy metrics, and displays accuracy curves across 10 training epochs with Matplotlib.

## Setup

Install the required Python dependencies:

```bash
pip install numpy librosa soundfile matplotlib tensorflow scikit-learn
```

The script runs on Python 3.9 or newer.

## Usage

Run the main pipeline:

```bash
python main.py
```

The script loads the audio files, computes features, fits the model, prints test metrics to standard output, and displays the training history plot.

## Configuration

Configure execution parameters directly in main.py:

* DATA_DIR: Set the absolute path to your LibriSpeech directory containing FLAC files and transcription files.
* max_files: Adjust the maximum number of audio files to process in the load_librispeech_subset function call. The default value is 20 files.
* test_size: Adjust the test partition fraction in train_test_split. The default value is 0.2 for an 80 percent training and 20 percent testing split.
* epochs: Set the number of training iterations in model.fit. The default value is 10 epochs.
* batch_size: Set the batch size for model updates. The default value is 32 samples.
