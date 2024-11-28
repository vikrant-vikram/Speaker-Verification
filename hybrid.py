import os
import time

import pandas as pd
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

import similarity_check as sc

import train_on_distance as tod

import gmm_modified as gmm
import gmm as gmm2
from collections import Counter


def get_most_frequent_value(values):
    """Return the value with the maximum frequency in a list."""
    if len(values) == 0:
        return None  # Handle empty input gracefully

    counter = Counter(values)
    most_common_value, _ = counter.most_common(1)[0]
    return most_common_value


def log_progress(message):
    """Print progress updates with timestamps."""
    print(f"\033[94m[Checkpoint]\033[0m [{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# # === Feature Extraction for SVM ===
# def extract_features(audio, sr=16000, n_mfcc=13):
#     """Extract various audio features compatible with the SVM model."""
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     mfcc_mean = np.mean(mfcc, axis=1)

#     pmfcc = librosa.feature.mfcc(y=librosa.effects.percussive(audio), sr=sr, n_mfcc=n_mfcc)
#     pmfcc_mean = np.mean(pmfcc, axis=1)

#     spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
#     spectral_centroid_mean = np.mean(spectral_centroid)

#     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
#     mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)

#     pitches, _ = librosa.piptrack(y=audio, sr=sr)
#     pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0

#     feature_vector = np.concatenate(
#             [mfcc_mean, pmfcc_mean, [spectral_centroid_mean], mel_spectrogram_mean, np.array([pitch_mean])]
#         )
#     return feature_vector


def extract_svm_features_segments(file_path, target_length=3, sr=16000, n_mfcc=13):
    """Extract SVM-compatible features from all segments of an audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        segment_length = target_length * sr
        if len(audio) < segment_length:
            audio = np.pad(audio, (0, segment_length - len(audio)))
        segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
        return [extract_features(segment, sr=sr, n_mfcc=n_mfcc) for segment in segments]
    except:
        return None


# === Extended Feature Extraction for Similarity Check ===
def extract_extended_features(audio, sr=16000, n_mfcc=13):
    """Extract extended features for similarity checks."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    pmfcc = librosa.feature.mfcc(y=librosa.effects.percussive(audio), sr=sr, n_mfcc=n_mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(pmfcc, axis=1),
        np.array([np.mean(spectral_centroid)]),
        np.mean(mel_spectrogram, axis=1),
        np.array([pitch_mean])
    ])


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
    try:
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

        # feature_vector = np.concatenate(
        #     [mfcc_mean, pmfcc_mean, [spectral_centroid_mean], mel_spectrogram_mean, [pitch_mean]]
        # )

        feature_vector = np.concatenate(
                [mfcc_mean, pmfcc_mean, [spectral_centroid_mean], mel_spectrogram_mean, np.array([pitch_mean])]
            )
        return feature_vector
    except:
        return None



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
    """Train the SVM model and save it."""
    metadata = get_path_to_wave(base_path)
    log_progress(f"Found {len(metadata)} files for training.")

    X, y = prepare_data(metadata, target_length=target_length, checkpoint_path=checkpoint_path)
    log_progress(f"Extracted {len(X)} feature vectors.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_progress("Split data into training and testing sets.")

    log_progress("Training SVM model...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    with open(model_path, "wb") as model_file:
        pickle.dump(svm, model_file)
    log_progress(f"Model saved to {model_path}")

    y_pred = svm.predict(X_test)
    print("Training complete. Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# === Verifying Speakers ===
def verify_speakers(model_path, file_path_1, file_path_2, target_length=3,
                     similarity_threshold=0.7):
    """Verify if two audio files are from the same speaker."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features_1 = extract_svm_features_segments(file_path_1, target_length=target_length)
    features_2 = extract_svm_features_segments(file_path_2, target_length=target_length)

    if features_1 is None or features_2 is None:
        log_progress("Error extracting features from one or both audio files.")

        return False

    predictions_1 = [model.predict([f])[0]for f in features_1]
    predictions_2 = [model.predict([f])[0]for f in features_2]

    user_1 = max(set(predictions_1), key=predictions_1.count)
    user_2 = max(set(predictions_2), key=predictions_2.count)

    if user_1 == user_2:
        print("----------model1------------")
        log_progress(f"Same speaker detected: User ID {user_1}")
        return True
    else:
        with open("svm_new.pkl", "rb") as f:
            model = pickle.load(f)

        features_1 = extract_svm_features_segments(file_path_1, target_length=target_length)
        features_2 = extract_svm_features_segments(file_path_2, target_length=target_length)

        if features_1 is None or features_2 is None:
            log_progress("Error extracting features from one or both audio files.")

            return False

        predictions_1 = [model.predict([f])[0]for f in features_1]
        predictions_2 = [model.predict([f])[0]for f in features_2]

        user_1 = max(set(predictions_1), key=predictions_1.count)
        user_2 = max(set(predictions_2), key=predictions_2.count)
        return user_1 == user_2


    # else:
    #     print("--------------model2 ----------------")
    #     l = []
    #     l.append(user_1 == user_2)
    #     l.append(gmm.verify_speakers(file_path_1, file_path_2, "enhanced_combined_model.pkl", sr=22050))
    #     l.append(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050))

    #     with open("speaker_verification_svm_optimized_on_distance.pkl", "rb") as f:
    #         model = pickle.load(f)
    #         features1 = tod.extract_features(file_path_1)
    #         features2 = tod.extract_features(file_path_2)
    #         feature_vector = np.array(tod.compute_distance_features(features1, features2)).reshape(1, -1)
    #         prediction = model.predict(feature_vector)[0]
    #         probability = model.predict_proba(feature_vector)[0][1]
    #         print("==================================with distance model==================================")
    #         print(prediction, probability)
    #         if(prediction and probability > 0.5 ):
    #             l.append(True)
    #         else:
    #             l.append(False)
    #     print(l, get_most_frequent_value(l))

    #     return get_most_frequent_value(l)
    # else:






    print("--------------model2 ----------------")
    l = []
    # l.append(user_1 == user_2)
    l.append(gmm.verify_speakers(file_path_1, file_path_2, "enhanced_combined_model.pkl", sr=22050))
    if(gmm.verify_speakers(file_path_1, file_path_2, "enhanced_combined_model.pkl", sr=22050)==True):
        l.append(gmm.verify_speakers(file_path_1, file_path_2, "enhanced_combined_model.pkl", sr=22050))
    l.append(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050))

    if(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050)==True):
        l.append(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050))

    # if(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050)==True):
    #     l.append(gmm2.verify_speakers(file_path_1, file_path_2, "combined_model_gmm.pkl", sr=22050))

    with open("speaker_verification_svm_optimized_on_distance.pkl", "rb") as f:
        model = pickle.load(f)
        features1 = tod.extract_features(file_path_1)
        features2 = tod.extract_features(file_path_2)
        feature_vector = np.array(tod.compute_distance_features(features1, features2)).reshape(1, -1)
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0][1]
        print("==================================with distance model==================================")
        print(prediction, probability)
        if(prediction and probability > 0.5 ):
            l.append(True)
        else:
            l.append(False)
    print(l, get_most_frequent_value(l))

    return get_most_frequent_value(l)



        # with open("speaker_verification_svm_optimized_on_distance.pkl", "rb") as f:
        #     model = pickle.load(f)
        #     features1 = tod.extract_features(file_path_1)
        #     features2 = tod.extract_features(file_path_2)
        #     feature_vector = np.array(tod.compute_distance_features(features1, features2)).reshape(1, -1)
        #     prediction = model.predict(feature_vector)[0]
        #     probability = model.predict_proba(feature_vector)[0][1]
        #     print("==================================with distance model==================================")
        #     print(prediction, probability)
        #     if(prediction and probability > 0.5 ):
        #         return True
        #     return prediction






def normalize_features(features):
    """Normalize features to zero mean and unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(features.reshape(-1, 1)).flatten()


def check_similarity(file_path_1, file_path_2, target_length=3):
    """Calculate similarity between two audio files."""
    audio_1, sr = librosa.load(file_path_1, sr=None)
    audio_2, _ = librosa.load(file_path_2, sr=int(sr))

    features_1 = extract_extended_features(audio_1, sr=int(sr))
    features_2 = extract_extended_features(audio_2, sr=int(sr))

    # Normalize feature vectors
    features_1 = normalize_features(features_1)
    features_2 = normalize_features(features_2)

    # Calculate similarity using cosine distance
    similarity = 1 - cosine(features_1, features_2)
    log_progress(f"Similarity score: {similarity}")

    return similarity



# === Helper Functions ===
def get_metadata(base_path):
    """Retrieve metadata from dataset directory."""
    files, users, paths = [], [], []
    for user_id in os.listdir(base_path):
        user_path = os.path.join(base_path, user_id)
        if os.path.isdir(user_path):
            for file in os.listdir(user_path):
                files.append(file)
                users.append(user_id)
                paths.append(os.path.join(user_path, file))
    return pd.DataFrame({"fileName": files, "userId": users, "Path": paths})


def handler(test_file_1, test_file_2):
    base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    features_checkpoint_path = "features_data_checkpoint.csv"
    model_path = "svm.pkl"
    if os.path.exists(model_path):
        log_progress("A trained model already exists.")
        # user_choice = input("Do you want to use the existing model for prediction (y) or retrain the model (n)? [y/nhghggh]: ").strip().lower()
        user_choice = 'y'
        if user_choice == 'y':

            log_progress("Starting speaker verification using the existing model.")
            return verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
        elif user_choice == 'nhghggh':
            log_progress("Retraining the model...")
            train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
            log_progress("Starting speaker verification with the retrained model.")
            return verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
        else:
            print("Invalid choice. Exiting.")
    else:
        log_progress("No trained model found. Starting training process.")
        train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
        log_progress("Starting speaker verification with the newly trained model.")
        return verify_speakers(model_path, test_file_1, test_file_2, target_length=3)




if __name__ == "__main__":
    # base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    # model_path = "svm.pkl"
    # features_checkpoint_path = "features_data_checkpoint.csv"
    # test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
#     features_checkpoint_path = "features_data_checkpoint.csv"
    # test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"

    test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10305/t0EX2dOBQjw/00004.wav"
    test_file_2 = "/Users/bikrant-bikram/Downloads/wav/id10305/ZLzkvnq0JxI/00002.wav"
    #
    # test_file_2 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
    #
    handler(test_file_1, test_file_2)


# if __name__ == "__main__":
#     base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
#     model_path = "svm.pkl"
#     features_checkpoint_path = "features_data_checkpoint.csv"
#     # test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"

#     test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10305/t0EX2dOBQjw/00004.wav"
#     test_file_2 = "/Users/bikrant-bikram/Downloads/wav/id10305/ZLzkvnq0JxI/00002.wav"
#     #
#     # test_file_2 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
#     #
#     #
#     if os.path.exists(model_path):
#         log_progress("A trained model already exists.")
#         user_choice = input("Do you want to use the existing model for prediction (y) or retrain the model (n)? [y/nhghggh]: ").strip().lower()
#         if user_choice == 'y':

#             log_progress("Starting speaker verification using the existing model.")
#             verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
#         elif user_choice == 'nhghggh':
#             log_progress("Retraining the model...")
#             train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
#             log_progress("Starting speaker verification with the retrained model.")
#             verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
#         else:
#             print("Invalid choice. Exiting.")
#     else:
#         log_progress("No trained model found. Starting training process.")
#         train_model(base_path, model_path, target_length=3, checkpoint_path=features_checkpoint_path)
#         log_progress("Starting speaker verification with the newly trained model.")
#         verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
