import os
import librosa
import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

from speaker_verification_eda import get_path_to_wave

# === Utility Functions ===
def log_progress(message):
    """Print progress updates with a visual separator."""
    print("\033[94m[Checkpoint]\033[0m", message)


def get_duration(file_path):
    """Get the duration of an audio file."""
    return librosa.get_duration(filename=file_path)



# === Audio Processing ===
def process_audio(file_path, sr=22050):
    """Load an audio file and return its waveform."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


# === Feature Extraction ===
def extract_mfcc_features(audio, sr=22050, n_mfcc=13):
    """Extract MFCC features with deltas and delta-deltas."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0).T


# === Model Training ===
def train_combined_model(metadata, n_components=16, output_model="combined_model.pkl"):
    """
    Train Gaussian Mixture Models (GMM) for each speaker and save them in one combined file.
    """
    log_progress("Training models for each speaker...")
    combined_model = {}

    for speaker in metadata["userId"].unique():
        speaker_data = metadata[metadata["userId"] == speaker]
        features = []

        for _, row in speaker_data.iterrows():
            audio = process_audio(row["Path"])
            features.append(extract_mfcc_features(audio))

        features = np.vstack(features)
        model = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=42)
        model.fit(features)

        combined_model[speaker] = model
        log_progress(f"Model trained for speaker: {speaker}")

    with open(output_model, "wb") as f:
        pickle.dump(combined_model, f)
    log_progress(f"Combined model saved to {output_model}")


# === Speaker Verification ===
def verify_speakers(file1, file2, model_path, sr=22050):
    """
    Verify if two audio files belong to the same speaker.
    """
    log_progress("Loading combined model...")
    with open(model_path, "rb") as f:
        combined_model = pickle.load(f)

    log_progress("Extracting features for Speaker 1...")
    audio1 = process_audio(file1, sr=sr)
    features1 = extract_mfcc_features(audio1, sr=sr)

    log_progress("Extracting features for Speaker 2...")
    audio2 = process_audio(file2, sr=sr)
    features2 = extract_mfcc_features(audio2, sr=sr)

    scores1 = {speaker: model.score(features1) for speaker, model in combined_model.items()}
    scores2 = {speaker: model.score(features2) for speaker, model in combined_model.items()}

    speaker1 = max(scores1, key=lambda k: scores1[k])
    speaker2 = max(scores2, key=lambda k: scores2[k])

    print(f"Speaker 1 ID: {speaker1}")
    print(f"Speaker 2 ID: {speaker2}")
    if speaker1 == speaker2:
        print("Same Speaker")
        return True

    else:
        return False


# === Main Execution ===
if __name__ == "__main__":
    BASE_PATH = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    MODEL_PATH = "combined_model_gmm.pkl"
    TEST_FILE_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
    TEST_FILE_2 ="/Users/bikrant-bikram/Downloads/wav/id10301/AeRSD9jIdSg/00006.wav"
    # base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    # model_path = "svm.pkl"
    # features_checkpoint_path = "features_data_checkpoint.csv"
    # test_file_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
    # test_file_2 = "/Users/bikrant-bikram/Downloads/wav/id10301/AeRSD9jIdSg/00006.wav"

    if os.path.exists(MODEL_PATH):
        log_progress("Using existing combined model for verification.")
        verify_speakers(TEST_FILE_1, TEST_FILE_2, MODEL_PATH)
    else:
        log_progress("Training a new combined model...")
        metadata = get_path_to_wave(BASE_PATH)
        train_combined_model(metadata, output_model=MODEL_PATH)

        log_progress("Starting verification with the newly trained model.")
        verify_speakers(TEST_FILE_1, TEST_FILE_2, MODEL_PATH)
