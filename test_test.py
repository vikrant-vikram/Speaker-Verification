import os
import pandas as pd
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# === STEP 1: Process Variable-Length Audio ===
def process_audio_segments(file_path, target_length=3, sr=16000):
    """
    Split long audio into segments of target length and pad shorter ones.
    Returns a list of audio segments.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    target_length_samples = target_length * sr  # Target length in samples
    segments = []

    # If the audio is longer, split into fixed-sized segments
    if len(audio) > target_length_samples:
        for start in range(0, len(audio), target_length_samples):
            segment = audio[start:start + target_length_samples]
            # Pad the last segment if it is shorter than target length
            if len(segment) < target_length_samples:
                segment = np.pad(segment, (0, target_length_samples - len(segment)))
            segments.append(segment)
    else:
        # Pad the audio if it is shorter
        audio = np.pad(audio, (0, target_length_samples - len(audio)))
        segments.append(audio)

    return segments

# === STEP 2: Extract MFCC Features ===
def extract_mfcc_segments(file_path, target_length=3, sr=16000, n_mfcc=13):
    """
    Extract MFCC features from audio segments.
    Returns a list of MFCC feature vectors.
    """
    audio_segments = process_audio_segments(file_path, target_length=target_length, sr=sr)
    mfcc_features = [np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).T, axis=0)
                     for segment in audio_segments]
    return mfcc_features

# === STEP 3: Load Metadata ===
def get_path_to_wave(base_path):
    """Load metadata of the dataset."""
    fileName = []
    userId = []
    path = []

    files_at_root = os.listdir(base_path)
    for id in files_at_root:
        if not os.path.isdir(os.path.join(base_path, id)):
            continue
        files_under_id = os.listdir(os.path.join(base_path, id))
        for sub_folders in files_under_id:
            if not os.path.isdir(os.path.join(base_path, id, sub_folders)):
                continue
            wave_files_in_sub_folder = os.listdir(os.path.join(base_path, id, sub_folders))
            for wave_file in wave_files_in_sub_folder:
                total_path = os.path.join(base_path, id, sub_folders, wave_file)
                fileName.append(wave_file)
                userId.append(id)
                path.append(total_path)

    df = pd.DataFrame({"fileName": fileName, "userId": userId, "Path": path})
    return df

# === STEP 4: Prepare Data ===
def prepare_data(metadata, target_length=3):
    """Prepare data by extracting MFCC features and labels."""
    features = []
    labels = []
    for idx, row in metadata.iterrows():
        mfcc_features = extract_mfcc_segments(row["Path"], target_length=target_length)
        for mfcc in mfcc_features:
            features.append(mfcc)
            labels.append(row["userId"])  # Each segment inherits the same label
    return np.array(features), np.array(labels)

# === STEP 5: Train Model ===
def train_model(base_path, model_path, target_length=3):
    """Train the SVM model and save it."""
    # Load metadata
    metadata = get_path_to_wave(base_path)

    # Prepare data
    X, y = prepare_data(metadata, target_length=target_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an SVM classifier
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, "wb") as model_file:
        pickle.dump(svm, model_file)

    # Test the model
    y_pred = svm.predict(X_test)
    print("Training complete. Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# === STEP 6: Verify Speakers ===
def verify_speakers(model_path, file_path_1, file_path_2, target_length=3):
    """Verify if two audio files are from the same speaker."""
    # Load the trained model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Extract MFCC features for both files
    mfcc_1 = extract_mfcc_segments(file_path_1, target_length=target_length)
    mfcc_2 = extract_mfcc_segments(file_path_2, target_length=target_length)

    # Predict user IDs for each segment and take the majority vote
    predictions_1 = [model.predict([mfcc])[0] for mfcc in mfcc_1]
    predictions_2 = [model.predict([mfcc])[0] for mfcc in mfcc_2]

    # Use majority voting to determine the final speaker ID
    user_1 = max(set(predictions_1), key=predictions_1.count)
    user_2 = max(set(predictions_2), key=predictions_2.count)

    print(f"Speaker 1 ID: {user_1}")
    print(f"Speaker 2 ID: {user_2}")
    if user_1 == user_2:
        print("Same Speaker")
    else:
        print("Different Speakers")

# === STEP 7: Main Function ===
if __name__ == "__main__":
    base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"  # Replace with your dataset directory
    model_path = "speaker_verification_model.pkl"  # Path to save the trained model

    # Train and save the model
    train_model(base_path, model_path, target_length=3)

    # # Test speaker verification
    # test_file_1 = "/path/to/test/audio1.wav"  # Replace with actual audio paths
    # test_file_2 = "/path/to/test/audio2.wav"
    # verify_speakers(model_path, test_file_1, test_file_2, target_length=3)
