#!/usr/bin/env python
# coding: utf-8

# ## **Voice-Based Cognitive Decline Pattern Detection**
# by Rishi Gupta
# ### **Introduction**
# This project deals with building a basic proof-of-concept pipeline that uses raw voice data samples to detect
# cognitive stress or decline indicators. We use techniques like audio processing, natural language processing along with suitable feature extraction.

# In[1]:


# Install and load modules
#get_ipython().system('pip install -q datasets pandas numpy matplotlib librosa pydub setuptools-rust openai-whisper')

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import whisper
import os
import pickle
import librosa
from pydub import AudioSegment
from pydub.silence import detect_silence


# For this task we are considering a audio dataset, obtained from the sources of the 'ADReSS Challenge' by 'DementiaBank'. We can perform deep audio analysis, along with transcribing to make suitable classifications.
# 
# The sample dataset consists of 30 audio files which have already been classified into 'dementia' and 'control'. This will be useful to train our model.

# In[2]:


# Transcribe audio files
AUDIO_FOLDER = "audio_data"

# We have moved all audio files from speaker folders to their parent folders, and selected 15 random samples manually

def transcribe_file(full_path, filename = None, subfolder = None):
    wp_model = whisper.load_model("tiny") # OpenAI's Whisper model for speech-to-text
    try:
        result = wp_model.transcribe(full_path)
        transcript = result["text"]
        return filename, transcript, subfolder
    
    except Exception as e:
        transcript = ''
        return filename, transcript, subfolder
    
def analyze_pauses(audio_file, min_silence_len=400, silence_threshold=-45):
    audio = AudioSegment.from_file(audio_file)
    
    silent_segments = detect_silence(audio, min_silence_len, silence_threshold)
    
    # Convert to seconds
    silent_segments = [(start/1000, stop/1000) for start, stop in silent_segments]
    
    # Calculate statistics
    pause_durations = [stop - start for start, stop in silent_segments]
    total_pauses = len(pause_durations)
    audio_duration = len(audio) / 1000.0
    
    # Pause rate (pauses per second)
    pause_rate = total_pauses / audio_duration if audio_duration > 0 else 0
    
    # Average pause duration
    avg_pause_duration = float(np.mean(pause_durations) if pause_durations else 0)
    
    # Extract just the filename from the full path
    filename = os.path.basename(audio_file)
    
    return {
        'filename': filename,
        'pause_rate': pause_rate,
        'avg_pause_duration': avg_pause_duration,
    }

import re
def detect_hesitations(transcript):
    # Define hesitation markers
    hesitation_markers = ['uh', 'um', 'er', 'ah', 'like', 'you know', 'hmm']
    
    # Count total hesitations
    total_hesitations = 0
    for marker in hesitation_markers:
        pattern = r'\b' + re.escape(marker) + r'\b'
        total_hesitations += len(re.findall(pattern, transcript.lower()))
    
    words = len(transcript.split())
    hesitation_rate = total_hesitations / words if words > 0 else 0
    
    return {
        'hesitation_rate': hesitation_rate
    }

def analyze_speech_acoustics(audio_file, sr=None):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr)
    
    # Extract pitch (F0) using librosa
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    
    # Remove NaN values from pitch
    f0_cleaned = f0[~np.isnan(f0)]
    
    # Calculate pitch statistics
    pitch_mean = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0
    pitch_std = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0
    pitch_range = np.ptp(f0_cleaned) if len(f0_cleaned) > 0 else 0
    
    # Estimate speech rate (syllables per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    speech_rate = tempo / 60
    
    return {
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'pitch_variability': pitch_std / pitch_mean if pitch_mean > 0 else 0,
        'pitch_range': pitch_range,
        'speech_rate': speech_rate,
    }

###########

if __name__ == "__main__":
    def transcribe_audio_folder(folder_path):
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif', '.m4a']
        
        # Find all audio files in the folder and subfolders
        audio_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, folder_path)
                    subfolder = "root" if rel_path == "." else rel_path
                    audio_files.append((full_path, file, subfolder))
                    
        results = []
        for full_path, filename, subfolder in audio_files:
            # print(f"Processing: {filename} from {subfolder}")

            filename, transcript, subfolder = transcribe_file(full_path, filename, subfolder)
            results.append({
                "filename": filename,
                "transcript": transcript,
                "output": subfolder
            })

        
        df = pd.DataFrame(results)
        return df

    # Create and display the dataframe with all the transcipts
    transcription_df = transcribe_audio_folder(AUDIO_FOLDER)


    # In[3]:


    transcription_df.head(10)


    # We have the necessary transcripts for the audio files, neatly catergorized with the existing labels. Now, we can focus on the deep audio analysis.
    # 
    # Let's start off by looking for pauses. Abrupt and long pauses are key signs of cognitive impairment. We will be looking for number of pauses, the rate of pauses and average pause times to make our assumptions.

    # In[4]:



    # Analyze all audio files and create a list of results
    pause_analysis_results = []
    for root, _, files in os.walk(AUDIO_FOLDER):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                full_path = os.path.join(root, file)
                pause_analysis_results.append(analyze_pauses(full_path))

    # Create dataframe and merge with exsisting dataframe
    pause_analysis_df = pd.DataFrame(pause_analysis_results)
    merged_df = pd.merge(transcription_df, pause_analysis_df, on='filename', how='left')


    # In[5]:


    merged_df.head(10)


    # The pause times vary, and by theory, we can typically determine the speech type based on the range of pause durations
    # - Healthy speech: 400–600 ms
    # - MCI: 600–800 ms
    # - Severe impairment: ≥ 800 ms

    # We now look for hesitation marks. The use of filler words like "um", "uh", "like" give us a sense of hesitation in the speech. Excessive use of these words can be a sign of impairment. We will again consider the rate of occurrance of these words as the lenghts of the audio files differ.

    # In[6]:



        
    # Analyze all transcripts in the dataframe
    hesitation_analysis_results = []
    for index, row in merged_df.iterrows():
        filename = row['filename']
        transcript = row['transcript']
        
        hesitation_data = detect_hesitations(transcript)
        
        # Add filename to the results
        hesitation_data['filename'] = filename
        
        # Append to results list
        hesitation_analysis_results.append(hesitation_data)

    # Create dataframe and merge with exsisting dataframe
    hesitation_analysis_df = pd.DataFrame(hesitation_analysis_results)

    merged2_df = pd.merge(merged_df, hesitation_analysis_df, on='filename', how='left')
    merged2_df.head(10)


    # We can analyse that the hesitation metrics are not as clean as we expected. This can be because the hesitation words may not be easily picked up by the transcription models, and it is not easy to properly determine them.

    # Now we move on to some more audio analysis, particularly speech rate and pitch variability.

    # In[7]:




    def analyze_audio_folder(folder_path):
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif', '.m4a']
        
        # Find all audio files in the folder and subfolders
        audio_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, folder_path)
                    subfolder = "root" if rel_path == "." else rel_path
                    audio_files.append((full_path, file, subfolder))
        
        # Analyze all audio files and create a list of results
        results = []
        for full_path, filename, subfolder in audio_files:
            # print(f"Analyzing acoustics: {filename} from {subfolder}")
            
            try:
                # Analyze speech acoustics
                acoustics_data = analyze_speech_acoustics(full_path)
                
                # Add filename and subfolder to the results
                acoustics_data['filename'] = filename
                acoustics_data['output'] = subfolder
                
                # Append to results list
                results.append(acoustics_data)
                
            except Exception as e:
                print(f"Error analyzing {filename}: {str(e)}")
                # Add a row with the filename but NaN values for metrics
                results.append({
                    'filename': filename,
                    'output': subfolder,
                    'pitch_mean': 0,
                    'pitch_std': 0,
                    'pitch_variability': 0,
                    'pitch_range': 0,
                    'speech_rate': 0
                })
        
        # Create dataframe with all the acoustic analysis results
        acoustics_df = pd.DataFrame(results)
        return acoustics_df

    # Create and analyze the acoustic features
    acoustics_df = analyze_audio_folder(AUDIO_FOLDER)

    # Merge with existing dataframe
    merged3_df = pd.merge(merged2_df, acoustics_df, on=['filename', 'output'], how='left')


    # In[8]:


    merged3_df.head(10)


    # Now that we have our final dataset, we will perform some cleaning and pre-processing and prepare the final dataset to conduct analysis and build and train our ML model on.

    # In[9]:


    # Move output column to the end
    col_to_move = 'output'
    merged3_df = merged3_df[[col for col in merged3_df.columns if col != col_to_move] + [col_to_move]]

    # Fix the speech_rate column - handle square brackets and convert to float
    def extract_float_from_brackets(value):
        try:
            # Remove brackets and convert to float
            stripped = str(value).strip('[]')
            return float(stripped)
        except (ValueError, TypeError):
            print(f"Conversion failed for value: '{value}'")
            return np.nan

    # Apply the conversion function
    merged3_df['speech_rate'] = merged3_df['speech_rate'].apply(extract_float_from_brackets)


    # In[10]:


    print(merged3_df.head(10))


    # We now focus on analysing the data. Plotting the feature distributions by class and doing a correlation analysis can be a good starting point

    # In[11]:


    import seaborn as sns
    from scipy import stats

    # Load the data
    data = merged3_df.copy()

    # Create feature matrix X and target vector y
    X = data[['pause_rate', 'avg_pause_duration', 'hesitation_rate', 
            'pitch_mean', 'pitch_std', 'pitch_variability', 
            'pitch_range', 'speech_rate']]
    y = data['output']

    # Plot feature distributions by class
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(X.columns):
        plt.subplot(3, 3, i+1)
        for target_class in y.unique():
            sns.kdeplot(X[feature][y == target_class], label=target_class)
        plt.title(f'Distribution of {feature}')
        plt.legend()
    plt.tight_layout()

    # Run t-tests between control and dementia groups for each feature
    print("\nT-test Results (Control vs Dementia):")
    print("-" * 60)
    print(f"{'Feature':<20} {'t-statistic':<15} {'p-value':<15} {'Significant':<10}")
    print("-" * 60)

    for feature in X.columns:
        control_data = X[feature][y == 'control']
        dementia_data = X[feature][y == 'dementia']
        
        t_stat, p_val = stats.ttest_ind(control_data, dementia_data, equal_var=False)
        significant = "Yes" if p_val < 0.05 else "No"
        
        # Print results
        print(f"{feature:<20} {t_stat:15.4f} {p_val:15.4f} {significant:<10}")


    # After performing t-test, we find out that none of the statistics are significant. This is not of concern, as although each feature individually does not contributing in classifying our samples, the combination of these features could help us in that task.
    # 
    # We first plot a correlation matrix to determine what combination of features is suited the best, and then we can move on the building and training a classifier model.

    # In[12]:


    # Feature correlation analysis
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()


    # Some key observations from the correlation analysis are:
    # 
    # - Strong positive correlations exist between pitch-related features, specifically `pitch_mean`, `pitch_std`, and `pitch_range` (correlations up to 0.86).
    # - Moderate positive correlations are seen between pause metrics (`pause_rate`, `avg_pause_duration`) and `pitch_range`.
    # - Strong negative correlations are observed between `hesitation_rate` and `pitch_range` (-0.36).
    # - `speech_rate` shows generally weak correlations with most other features.

    # Let's move on to building and training the model. We will build a Random Forest model to classify the data, evaluate its performance, and fine-tune it using hyperparameter optimization techniques.

    # In[13]:


    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report


    X = data.drop(['output', 'filename', 'transcript'], axis=1)
    y = data['output']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test_scaled)
    print("\nRandom Forest Model Results:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Leave One Out Cross Validation
    loo = LeaveOneOut()
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=loo, scoring='accuracy')
    print(f"\nLeaveOneOut CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    # Get top 3 features based on importance
    top_features = feature_importances.head(3)['Feature'].tolist()
    print(f"\nSelected top 3 features for optimization: {top_features}")

    # Create a dataset with only top features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    X_train_top_scaled = scaler.fit_transform(X_train_top)
    X_test_top_scaled = scaler.transform(X_test_top)

    # Optimize Model
    param_grid = {
        'n_estimators': [20, 50, 100],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3, 5]
    }

    # First optimize using all features
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=loo,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    print("\nBest parameters from GridSearchCV (using all features):")
    print(grid_search.best_params_)

    # Then optimize using only top features
    grid_search_top = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=loo,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_top.fit(X_train_top_scaled, y_train)
    print("\nBest parameters from GridSearchCV (using top features):")
    print(grid_search_top.best_params_)

    # Train model with best parameters
    best_rf_model = RandomForestClassifier(
        **grid_search.best_params_,
        random_state=42,
    )
    best_rf_model.fit(X_train_scaled, y_train)

    # Train model with top parameters
    best_rf_model_top = RandomForestClassifier(
        **grid_search_top.best_params_,
        random_state=42,
    )
    best_rf_model_top.fit(X_train_top_scaled, y_train)

    # Evaluate the models
    print("\nOptimized Random Forest Model Results (all features):")
    best_y_pred = best_rf_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, best_y_pred))
    optimized_cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=loo, scoring='accuracy')
    print(f"\nOptimized Model LOOCV accuracy (all features): {optimized_cv_scores.mean():.4f} (±{optimized_cv_scores.std():.4f})")

    print("\nOptimized Random Forest Model Results (top features only):")
    best_y_pred_top = best_rf_model_top.predict(X_test_top_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, best_y_pred_top))
    optimized_cv_scores_top = cross_val_score(best_rf_model_top, X_train_top_scaled, y_train, cv=loo, scoring='accuracy')
    print(f"\nOptimized Model LOOCV accuracy (top features): {optimized_cv_scores_top.mean():.4f} (±{optimized_cv_scores_top.std():.4f})")


    # The Random Forest model achieves strong performance on the held-out test set (accuracy: 0.78). After hyperparameter optimization, the test set accuracy further increases to 0.89 when using all features. However, Leave-One-Out Cross-Validation (LOOCV) reveals much lower and highly variable accuracy (0.43 ± 0.49), indicating that the model’s generalization to unseen data is likely poor and unstable—likely due to the very small dataset size.
    # 
    # Feature selection using only the top 3 most important features results in a lower test set accuracy (0.56), but a modest improvement in LOOCV accuracy (0.57 ± 0.49). This suggests that while the model with all features may overfit to the small test set, the model with fewer features may generalize slightly better, though overall performance remains limited by the dataset size.
    # 
    # In summary, the high test set scores likely overestimate true model performance. The LOOCV results provide a more realistic estimate of generalization. Increasing the dataset size is essential for building a more robust and reliable model, but for due high training time, we are limited to using a total of only 30 samples.
    # 

    # We can now save our model, and use it autonomously to make classification on samples outside the training data.

    # In[14]:


    # Save the best model and scaler
    model_selected = "top"

    if model_selected == "top":
        model_dir = f"saved_model_{model_selected}"
    else:
        model_dir = "saved_model_other"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model with all features (better test performance)
    with open(os.path.join(model_dir, f'best_rf_model_{model_selected}.pkl'), 'wb') as f:
        pickle.dump(best_rf_model_top, f)

    # Save the scaler
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature names for reference
    with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(list(X.columns), f)

    print(f"\nModel '{model_selected}' and scaler saved to {model_dir} directory")


    # 
    # ### **Conclusion**
    # 
    # - This report, where we performing deep analysis and building classification models, helps us understand the basic factors and metrics that contribute in determining cognitive stress or impairment in a person, by soley depending on audio recordings and features.
    # 
    # - Feature importance and correlation analysis reveals that pitch_std and pitch_variability are the most influential predictors, followed by pause-related features, and hesitation_rate had the least importance.
    # 
    # - We implement the Random Forest model for this project as it is great at classification tasks, and allows for hyperparameter tuning. It is successful in classifying the audio files, and the hyperparameter tuning is carried out based on the feature importances determined earlier.

    # In the `test_model.py` file, we can test the model we have just build on unseen audio samples.
