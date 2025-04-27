# Voice-Based Cognitive Pattern Detection

This project provides a pipeline to analyze raw voice audio samples and detect patterns associated with cognitive stress or decline, specifically focusing on distinguishing between 'control' and 'dementia' classifications based on the ADReSS Challenge dataset from DementiaBank. It utilizes audio processing, speech-to-text transcription, and machine learning techniques.

## Overview

The core idea is to extract various features from speech recordings that are potentially indicative of cognitive impairment. These features include:
-   **Audio Transcripts:** Converting speech to text using OpenAI's Whisper model 
*   **Pause Metrics:** Pause rate, average pause duration.
*   **Hesitation Metrics:** Rate of filler words (um, uh, etc.).
*   **Acoustic Features:** Pitch mean, pitch standard deviation, pitch variability, pitch range, speech rate.

These features are then used to train a Random Forest classifier to distinguish between audio samples labelled as 'control' and 'dementia'.

## File Structure

*   `notebook.ipynb` / `my_notebook.py`: Jupyter notebook (and its Python script version) containing the main workflow:
*   `test_model.py`: A Python script to load a pre-trained model and test it on a single, new audio file.
*   `saved_model_top/`: Directory where the trained model (`best_rf_model_top.pkl`), scaler (`scaler.pkl`), and feature names (`feature_names.pkl`) are stored.
*   `audio_data/` (Required): You need to create this directory and populate it with your audio dataset for training (see Dataset Preparation).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rishig10/Voice-Based-Pattern-Detection.git
    cd Voice-Based-Pattern-Detection
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib librosa pydub openai-whisper scikit-learn
    ```
    *   **Note:** OpenAI Whisper requires `ffmpeg` to be installed on your system. Please follow the installation instructions if you don't have it.

## Dataset Preparation

*   The training script expects audio files to be organized within an `audio_data` directory.
*   Inside `audio_data`, create subdirectories corresponding to your classes (e.g., `control`, `dementia`).
*   Place your `.wav`, `.mp3`, or other supported audio files into the respective class subdirectories.

    ```
    Voice-Based-Pattern-Detection/
    ├── audio_data/
    │   ├── control/
    │   │   ├── control_sample_1.wav
    │   │   └── ...
    │   └── dementia/
    │       ├── dementia_sample_1.wav
    │       └── ...
    ├── my_notebook.py
    ├── test_model.py
    ├── saved_model_top/
    ```
*   The original notebook used data derived from the ADReSS Challenge (DementiaBank), but you will need to provide your own dataset structured as described above.

## Usage

### 1. Training the Model

*   Ensure your dataset is prepared in the `audio_data` directory.
*   Run the `my_notebook.py` script (or execute the cells in `notebook.ipynb`):
    ```bash
    python my_notebook.py
    ```
*   This script will:
    *   Transcribe all audio files in `audio_data`.
    *   Extract all defined features.
    *   Perform data analysis and train a Random Forest model.
    *   Save the best model (`best_rf_model_top.pkl`), the fitted scaler (`scaler.pkl`), and the list of feature names (`feature_names.pkl`) into the `saved_model_top/` directory.

### 2. Testing the Model on a New Audio File

*   Make sure you have successfully run the training step first, so the `saved_model_top/` directory contains the necessary `.pkl` files.
*   Run the `test_model.py` script:
    ```bash
    python test_model.py
    ```
*   The script will prompt you to enter the full path to the audio file you want to classify.
*   It will then print the extracted features, the predicted class, and the model's confidence score.
