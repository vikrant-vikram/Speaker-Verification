
# import os
# import pandas as pd
# import librosa
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report

import os
import pandas as pd
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed
import multiprocessing

# === Utility Functions ===
def log_progress(message):
    """Print progress updates with a visual separator."""
    print("\033[94m[Checkpoint]\033[0m", message)


def save_partial_features(features, labels, checkpoint_path):
    """Save intermediate features and labels during feature extraction."""
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(checkpoint_path, index=False)
    log_progress(f"Saved intermediate features to {checkpoint_path}")


def load_partial_features(checkpoint_path):
    """Load intermediate features and labels."""
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        labels = df['label'].values
        features = df.drop(columns=['label']).values
        log_progress(f"Loaded intermediate features from {checkpoint_path}")
        return features, labels
    return None, None


# === Audio Processing ===
def process_audio_segments(file_path, target_length=3, sr=16000):
    """Split audio into fixed-length segments or pad if shorter."""
    audio, sample_rate = librosa.load(file_path, sr=sr)
    target_length_samples = target_length * sr
    segments = []

    if len(audio) > target_length_samples:
        for start in range(0, len(audio), target_length_samples):
            segment = audio[start:start + target_length_samples]
            if len(segment) < target_length_samples:
                segment = np.pad(segment, (0, target_length_samples - len(segment)))
            segments.append(segment)
    else:
        audio = np.pad(audio, (0, target_length_samples - len(audio)))
        segments.append(audio)

    return segments


# === Feature Extraction ===
def extract_features(audio, sr=16000, n_mfcc=13):
    """Extract various audio features."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    pmfcc = librosa.feature.mfcc(y=librosa.effects.percussive(audio), sr=sr, n_mfcc=n_mfcc)
    pmfcc_mean = np.mean(pmfcc, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)

    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0


    feature_vector = np.concatenate(
            [mfcc_mean, pmfcc_mean, [spectral_centroid_mean], mel_spectrogram_mean, np.array([pitch_mean])]
        )
    return feature_vector


def extract_features_segments(file_path, target_length=3, sr=16000, n_mfcc=13):
    """Extract features from all segments of an audio file."""
    audio_segments = process_audio_segments(file_path, target_length=target_length, sr=sr)
    features = [extract_features(segment, sr=sr, n_mfcc=n_mfcc) for segment in audio_segments]
    return features


# === Dataset Processing ===
def get_path_to_wave(base_path):
    """Get metadata for the dataset."""
    fileName = []
    userId = []
    duration = []
    path = []

    files_at_root = os.listdir(base_path)
    for id in files_at_root:
        if not os.path.isdir(os.path.join(base_path, id)):
            continue
        files_under_id = os.listdir(os.path.join(base_path, id))
        for sub_folder in files_under_id:
            if not os.path.isdir(os.path.join(base_path, id, sub_folder)):
                continue
            wave_files = os.listdir(os.path.join(base_path, id, sub_folder))
            for wave_file in wave_files:
                file_path = os.path.join(base_path, id, sub_folder, wave_file)
                fileName.append(wave_file)
                userId.append(id)
                duration.append(librosa.get_duration(filename=file_path))
                path.append(file_path)

    return pd.DataFrame({"fileName": fileName, "userId": userId, "Duration": duration, "Path": path})


def prepare_data(metadata, target_length=3, checkpoint_path="features_data_checkpoint.csv"):
    """Prepare data by extracting features and saving/loading checkpoints."""
    features, labels = load_partial_features(checkpoint_path)
    if features is not None and labels is not None:
        return features, labels

    log_progress("Starting feature extraction...")
    all_features = []
    all_labels = []

    for idx, row in metadata.iterrows():
        try:
            feature_vectors = extract_features_segments(row["Path"], target_length=target_length)
            for feature_vector in feature_vectors:
                all_features.append(feature_vector)
                all_labels.append(row["userId"])

            if idx % 10 == 0:
                save_partial_features(all_features, all_labels, checkpoint_path)
        except Exception as e:
            log_progress(f"Error processing file {row['Path']}: {e}")

    save_partial_features(all_features, all_labels, checkpoint_path)
    log_progress("Feature extraction completed.")
    return np.array(all_features), np.array(all_labels)



# === Model Training ===
def train_model(base_path, model_path, target_length=3, checkpoint_path="features_data_checkpoint.csv"):
    """Train the model using pre-saved features with parallelization and parameter tuning."""
    log_progress("Loading pre-saved features from checkpoint...")
    X, y = load_partial_features(checkpoint_path)

    if X is None or y is None:
        log_progress("No pre-saved features found. Extracting features...")
        metadata = get_path_to_wave(base_path)
        X, y = prepare_data(metadata, target_length=target_length, checkpoint_path=checkpoint_path)

    log_progress(f"Loaded {len(X)} feature vectors for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_progress("Data split into training and testing sets.")

    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    gmm_params = [{'n_components': n, 'covariance_type': cov} for n in [2, 4, 8, 16] for cov in ['full', 'diag']]

    models = {"SVM": SVC(probability=True), "GMM": GaussianMixture()}

    def train_model_with_params(model_name, model, param_grid):
        """Train a specific model with parameter tuning."""
        log_progress(f"Training {model_name}...")
        if model_name == "GMM":
            best_model, best_score = None, -np.inf
            for params in param_grid:
                gmm = GaussianMixture(**params).fit(X_train)
                score = gmm.score(X_train)
                if score > best_score:
                    best_model, best_score = gmm, score
            return model_name, best_model
        else:
            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            return model_name, grid_search.best_estimator_

    num_cores = multiprocessing.cpu_count()
    trained_models = Parallel(n_jobs=num_cores)(
        delayed(train_model_with_params)(name, model, svm_params if name == "SVM" else gmm_params)
        for name, model in models.items()
    )

    best_model_name, best_model, best_accuracy = None, None, -np.inf
    for name, model in trained_models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) if name == "SVM" else np.nan
        print(f"{name} Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_model_name, best_model = name, model

    with open(model_path, "wb") as model_file:
        pickle.dump(best_model, model_file)
    log_progress(f"Best model ({best_model_name}) saved to {model_path}")



# === Speaker Verification ===
def verify_speakers(model_path, file_path_1, file_path_2, target_length=3):
    """Verify if two speakers are the same."""
    log_progress("Loading model for verification...")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    log_progress("Model loaded.")

    log_progress("Extracting features from speaker 1...")
    features_1 = extract_features_segments(file_path_1, target_length=target_length)
    log_progress("Extracting features from speaker 2...")
    features_2 = extract_features_segments(file_path_2, target_length=target_length)

    predictions_1 = [model.predict([feature])[0] for feature in features_1]
    predictions_2 = [model.predict([feature])[0] for feature in features_2]

    user_1 = max(set(predictions_1), key=predictions_1.count)
    user_2 = max(set(predictions_2), key=predictions_2.count)

    print(f"Speaker 1 ID: {user_1}")
    print(f"Speaker 2 ID: {user_2}")
    if user_1 == user_2:
        print("Same Speaker")
    else:
        print("Different Speakers")


if __name__ == "__main__":
    base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    model_path = "svm.pkl"
    features_checkpoint_path = "features_data_checkpoint.csv"
    test_file_1 = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/id10270/5r0dWxy17C8/00001.wav_chunk_1.wav"
    test_file_2 = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/id10270/5r0dWxy17C8/00004.wav_chunk_1.wav"
    if os.path.exists(model_path):
        log_progress("A trained model already exists.")
        user_choice = input("Do you want to use the existing model for prediction (y) or retrain the model (n)? [y/n]: ").strip().lower()
        if user_choice == 'y':

            log_progress("Starting speaker verification using the existing model.")
            verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
        elif user_choice == 'n':
            log_progress("Retraining the model...")
            train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
            log_progress("Starting speaker verification with the retrained model.")
            verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
        else:
            print("Invalid choice. Exiting.")
    else:
        log_progress("No trained model found. Starting training process.")
        train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
        log_progress("Starting speaker verification with the newly trained model.")
        verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
