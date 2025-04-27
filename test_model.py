TEST_AUDIO_PATH = input("Enter the path to the audio file you want to test: ")

from notebook_converted import analyze_pauses, analyze_speech_acoustics, detect_hesitations, transcribe_file
import pandas as pd
import pickle
import os
import numpy as np

# Load the trained model, scaler and training data features
model_path = f"saved_model_top/best_rf_model_top.pkl"
scaler_path = f"saved_model_top/scaler.pkl"
X_path = f"saved_model_top/feature_names.pkl"

with open(model_path, 'rb') as f:
    best_rf_model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

if os.path.exists(X_path):
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
        if isinstance(X, list):
            X = pd.DataFrame(columns=X)
else:
    X = None

# Get feature names from the scaler
scaler_feature_names = []
if hasattr(scaler, 'feature_names_in_'):
    scaler_feature_names = scaler.feature_names_in_.tolist()

# Test the model on a new audio file
def extract_features_from_audio(audio_file_path):
    filename, transcript, subfolder = transcribe_file(audio_file_path)
    pause_dict = analyze_pauses(audio_file_path)
    speech_acoustics = analyze_speech_acoustics(audio_file_path)
    
    pause_rate = pause_dict['pause_rate']
    avg_pause_duration = pause_dict['avg_pause_duration']
    hesitation_result = detect_hesitations(transcript)

    if isinstance(hesitation_result, dict) and 'hesitation_rate' in hesitation_result:
        hesitation_rate = hesitation_result['hesitation_rate']
    else:
        print(f"WARNING: Unexpected hesitation_result format: {type(hesitation_result)}")
        hesitation_rate = hesitation_result if isinstance(hesitation_result, (int, float)) else 0
    
    pitch_mean = speech_acoustics['pitch_mean']
    pitch_std = speech_acoustics['pitch_std']
    pitch_variability = speech_acoustics['pitch_variability']
    pitch_range = speech_acoustics['pitch_range']
    speech_rate = speech_acoustics['speech_rate']
    
    if isinstance(speech_rate, (list, np.ndarray)) and len(speech_rate) > 0:
        speech_rate_value = float(speech_rate[0])
    else:
        speech_rate_value = speech_rate
    
    # Create a feature dictionary
    features = {
        'pause_rate': pause_rate,
        'avg_pause_duration': avg_pause_duration,
        'hesitation_rate': hesitation_rate,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'pitch_variability': pitch_variability,
        'pitch_range': pitch_range,
        'speech_rate': speech_rate_value
    }
    
    # Check that all features are numeric
    for key, value in features.items():
        if not isinstance(value, (int, float)):
            print(f"Feature {key} is not numeric: {value} ({type(value)})")
            features[key] = 0  # Default to 0 if not numeric
    
    return features, transcript

# Test on a new audio file
try:
    test_features, test_transcript = extract_features_from_audio(TEST_AUDIO_PATH)
    test_df = pd.DataFrame([test_features])
    
    if isinstance(X, list):
        X = pd.DataFrame(columns=X)
    
    # Use feature names from scaler if available, otherwise use X
    feature_names_to_use = scaler_feature_names if scaler_feature_names else (X.columns.tolist() if hasattr(X, 'columns') else [])
    
    if feature_names_to_use:
        new_test_df = pd.DataFrame(index=test_df.index)
        
        for feature in feature_names_to_use:
            if feature in test_df.columns:
                new_test_df[feature] = test_df[feature]
            else:
                new_test_df[feature] = 0
        
        # Replace test_df with the properly structured one
        test_df = new_test_df
    else:
        print("Warning: No feature names available, using features as-is!")
        
    # Scale the features
    test_scaled = scaler.transform(test_df)
    
    print("\nExtracted Features:")
    for feature, value in test_features.items():
        print(f"{feature}: {value}")
    
    prediction = best_rf_model.predict(test_scaled)
    prediction_proba = best_rf_model.predict_proba(test_scaled)
    
    print("\nTest Audio Classification:")
    print(f"File: {TEST_AUDIO_PATH}")
    print(f"Transcript: {test_transcript}")
    print(f"Prediction: {prediction[0]}")
    print(f"Confidence: {max(prediction_proba[0]) * 100:.2f}%")
        
except Exception as e:
    print(f"Error testing on new audio: {str(e)}")