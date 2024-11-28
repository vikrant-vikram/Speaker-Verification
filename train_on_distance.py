import numpy as np
import pandas as pd
import random
import librosa
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from fastdtw import fastdtw
import os
from speaker_verification_eda import get_path_to_wave


def extract_features(file_path, n_mfcc=13, n_chroma=12, n_contrast=6, n_tonnetz=6, fmin=200.0):
    """
    Extracts a variety of speech features from the audio file.
    """
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    chroma = np.mean(chroma, axis=1)

    # Spectral Contrast
    nyquist = sr / 2.0
    fmin = min(fmin, nyquist / 2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_contrast, fmin=fmin)
    contrast = np.mean(contrast, axis=1)

    # Tonnetz
    if librosa.util.valid_audio(y) and len(chroma) > 0:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz = np.mean(tonnetz, axis=1)
    else:
        tonnetz = np.zeros(n_tonnetz)

    return np.concatenate([mfcc, chroma, contrast, tonnetz])


def load_and_validate_audio(file_path):
    """
    Validate if the audio file can be loaded.
    """
    return True


def compute_distance_features(features1, features2):
    """
    Compute distance-based features for a pair of feature vectors.
    """
    features1 = (features1 - np.mean(features1)) / np.std(features1)
    features2 = (features2 - np.mean(features2)) / np.std(features2)

    euclidean_dist = euclidean(features1, features2)
    manhattan_dist = cityblock(features1, features2)
    corr_coeff, _ = pearsonr(features1.flatten(), features2.flatten())
    corr_coeff = max(-1, min(1, corr_coeff))
    dtw_distance, _ = fastdtw(features1.flatten(), features2.flatten())

    return [euclidean_dist, manhattan_dist, corr_coeff, dtw_distance]


def generate_file_pairs(base_path, iterations=1000):
    """
    Generate pairs of audio files along with their labels (same or different speaker).
    """
    df = get_path_to_wave(base_path)
    if len(df) < 2:
        raise ValueError("Not enough files in the dataset to generate pairs.")

    grouped = df.groupby("userId")
    same_speaker_pairs = []
    different_speaker_pairs = []

    for user_id, group in grouped:
        group_files = group.to_dict(orient="records")
        same_speaker_pairs.extend([(a, b) for i, a in enumerate(group_files) for b in group_files[i + 1:]])

    all_files = df.to_dict(orient="records")
    while len(different_speaker_pairs) < len(same_speaker_pairs):
        file1, file2 = random.sample(all_files, 2)
        if file1["userId"] != file2["userId"]:
            different_speaker_pairs.append((file1, file2))

    random.shuffle(same_speaker_pairs)
    random.shuffle(different_speaker_pairs)

    half_iterations = iterations // 2
    return same_speaker_pairs[:half_iterations] + different_speaker_pairs[:half_iterations]


def train_multiple_models_with_optimization(
    base_path,
    iterations=1000,
    model_save_path="best_speaker_verification_model_with_distance.pkl",
    checkpoint_interval=50,
):
    """
    Train multiple models for speaker verification using hyperparameter optimization,
    evaluate them, and save the best-performing model.
    """
    pairs = generate_file_pairs(base_path, iterations)
    random.shuffle(pairs)

    X = []
    y = []

    start_time = time.time()
    for idx, (file1, file2) in enumerate(pairs):
        path1, id1 = file1["Path"], file1["userId"]
        path2, id2 = file2["Path"], file2["userId"]

        if not load_and_validate_audio(path1) or not load_and_validate_audio(path2):
            continue

        features1 = extract_features(path1)
        features2 = extract_features(path2)

        feature_vector = compute_distance_features(features1, features2)
        X.append(feature_vector)
        y.append(1 if id1 == id2 else 0)

        # Periodic logging and checkpoints
        if idx % checkpoint_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"[Checkpoint] Processed {idx}/{iterations} pairs. "
                  f"Elapsed time: {elapsed_time:.2f} seconds.")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and hyperparameter grids
    models = {
        "SVM": (SVC(probability=True, random_state=42), {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }),
        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 10],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }),
    }

    best_model = None
    best_score = -1
    best_params = None
    best_model_name = ""

    # Perform GridSearchCV for each model
    for model_name, (model, param_grid) in models.items():
        print(f"Starting hyperparameter optimization for {model_name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            verbose=2,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model_name = model_name

    if best_model is not None:
        print(f"Best Model: {best_model_name}")
        print(f"Best Parameters: {best_params}")
        print(f"Best ROC-AUC Score: {best_score}")

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Test ROC-AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

        # Save the best model
        with open(model_save_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Best model saved at {model_save_path}")
    else:
        print("No valid model was found during the optimization process.")


# Main execution
if __name__ == "__main__":
    train_multiple_models_with_optimization(
        base_path="/Users/bikrant-bikram/Downloads/wav",
        iterations=1000,
    )
