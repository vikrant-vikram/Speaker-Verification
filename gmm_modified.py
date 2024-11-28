import os
import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from speaker_verification_eda import get_path_to_wave

# === Utility Functions ===
def log_progress(message):
    """Print progress updates with a visual separator."""
    print("\033[94m[Checkpoint]\033[0m", message)


# === Audio Processing ===
def process_audio(file_path, sr=22050):
    """Load an audio file and return its waveform."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


# === Feature Extraction ===
def extract_mfcc_features(audio, sr=22050, n_mfcc=20):
    """Extract enhanced MFCC and spectral features."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc, spectral_centroid, spectral_bandwidth, chroma], axis=0)
    return features.T


# === Data Augmentation ===
# def augment_audio(audio, sr=22050):
#     """Augment audio with noise, speed, and pitch changes."""
#     noise = audio + 0.005 * np.random.randn(len(audio))
#     speed = librosa.effects.time_stretch(audio, rate=1.1)
#     pitch = librosa.effects.pitch_shift(audio, sr, n_steps=2)
#     return [audio, noise, speed, pitch]

def augment_audio(audio, sr=22050):
    """Augment audio with noise, speed, and pitch changes."""
    noise = audio + 0.005 * np.random.randn(len(audio))  # Add noise
    speed = librosa.effects.time_stretch(audio, rate=1.1)  # Speed change
    pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)  # Pitch change
    return [audio, noise, speed, pitch]


# === Model Training ===
def train_combined_model(metadata, n_components=16, output_model="enhanced_combined_model_gmm.pkl"):
    """
    Train Gaussian Mixture Models (GMM) for each speaker and save them in one combined file.
    """
    log_progress("Splitting data for unseen speaker testing...")
    speakers = metadata["userId"].unique()
    train_speakers, test_speakers = train_test_split(speakers, test_size=0.2, random_state=42)

    train_data = metadata[metadata["userId"].isin(train_speakers)]
    test_data = metadata[metadata["userId"].isin(test_speakers)]

    log_progress("Training models for each speaker...")
    combined_model = {}

    for speaker in train_data["userId"].unique():
        speaker_data = train_data[train_data["userId"] == speaker]
        features = []

        for _, row in speaker_data.iterrows():
            audio = process_audio(row["Path"])
            augmented_audios = augment_audio(audio)  # Data augmentation
            for aug_audio in augmented_audios:
                features.append(extract_mfcc_features(aug_audio))

        features = np.vstack(features)
        model = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=42)
        model.fit(features)

        combined_model[speaker] = model
        log_progress(f"Model trained for speaker: {speaker}")

    with open(output_model, "wb") as f:
        pickle.dump(combined_model, f)
    log_progress(f"Combined model saved to {output_model}")

    return train_data, test_data


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

    # Cosine Similarity
    log_progress("Computing similarity scores...")
    scores1 = {speaker: model.score(features1) for speaker, model in combined_model.items()}
    scores2 = {speaker: model.score(features2) for speaker, model in combined_model.items()}

    speaker1 = max(scores1, key=lambda k: scores1[k])
    speaker2 = max(scores2, key=lambda k: scores2[k])

    similarity = 1 - cosine(np.mean(features1, axis=0), np.mean(features2, axis=0))

    print(f"Speaker 1 ID: {speaker1}")
    print(f"Speaker 2 ID: {speaker2}")
    print(f"Cosine Similarity: {similarity:.4f}")
    if speaker1 == speaker2:
        print("Same Speaker")
        return True
    else:
        return False


# === Main Execution ===
if __name__ == "__main__":
    BASE_PATH = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"
    MODEL_PATH = "enhanced_combined_model.pkl"
    TEST_FILE_1 = "/Users/bikrant-bikram/Downloads/wav/id10301/6uqGkiR2oTI/00001.wav"
    TEST_FILE_2 = "/Users/bikrant-bikram/Downloads/wav/id10301/AeRSD9jIdSg/00006.wav"

    if os.path.exists(MODEL_PATH):
        log_progress("Using existing combined model for verification.")
        result= verify_speakers(TEST_FILE_1, TEST_FILE_2, MODEL_PATH)
        print(f"Verification Result: {'Same Speaker' if result else 'Different Speakers'}")
    else:
        log_progress("Training a new combined model...")
        metadata = get_path_to_wave(BASE_PATH)
        train_data, test_data = train_combined_model(metadata, output_model=MODEL_PATH)

        log_progress("Starting verification with the newly trained model.")
        result= verify_speakers(TEST_FILE_1, TEST_FILE_2, MODEL_PATH)
        print(f"Verification Result: {'Same Speaker' if result else 'Different Speakers'}")
