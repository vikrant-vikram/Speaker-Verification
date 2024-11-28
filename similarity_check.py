import librosa
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import librosa.display
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr

import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13, n_chroma=12, n_contrast=6, n_tonnetz=6, fmin=200.0):
    """
    Extracts a variety of speech features from the audio file.
    :param file_path: Path to the audio file.
    :param n_mfcc: Number of MFCC coefficients to extract.
    :param n_chroma: Number of chroma features to extract.
    :param n_contrast: Number of spectral contrast bands.
    :param n_tonnetz: Number of tonnetz features to extract.
    :param fmin: Minimum frequency for spectral contrast bands.
    :return: Extracted features (MFCCs, Chroma, Spectral Contrast, Tonnetz).
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)  # Mean of each coefficient

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    chroma = np.mean(chroma, axis=1)

    # Spectral Contrast
    # Ensure fmin is below Nyquist frequency
    nyquist = sr / 2.0
    if fmin >= nyquist:
        fmin = nyquist / 2  # Adjust fmin to be safely below Nyquist frequency
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_contrast, fmin=fmin)
    contrast = np.mean(contrast, axis=1)

    # Tonnetz
    # Tonnetz requires chroma features
    if librosa.util.valid_audio(y) and len(chroma) > 0:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz = np.mean(tonnetz, axis=1)
    else:
        tonnetz = np.zeros(n_tonnetz)  # Fallback if tonnetz cannot be calculated

    # Combine all features into a single feature vector
    features = np.concatenate([mfcc, chroma, contrast, tonnetz])

    return features


def compute_cosine_similarity(vector1, vector2):
    """
    Compute the cosine similarity between two vectors.
    :param vector1: First vector (1D numpy array).
    :param vector2: Second vector (1D numpy array).
    :return: Cosine similarity value.
    """
    # Ensure vectors are 1D
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()

    # Check if vectors are not zero-vectors
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        raise ValueError("One or both vectors are zero vectors, cannot compute cosine similarity.")

    # Compute cosine similarity
    similarity = 1 - cosine(vector1, vector2)
    return similarity



# def compare_features(features1, features2, threshold=0.01):
#     """
#     Compare two feature vectors using multiple distance metrics.
#     :param features1: Features of the first audio file.
#     :param features2: Features of the second audio file.
#     :param threshold: Cosine similarity threshold to classify as same or different speaker.
#     :return: True if both features indicate the same speaker, False otherwise.
#     """
#     # Normalize features to zero mean and unit variance
#     features1 = (features1 - np.mean(features1)) / np.std(features1)
#     features2 = (features2 - np.mean(features2)) / np.std(features2)

#     # Ensure PCA is only applied if there are enough dimensions
#     n_samples = 2  # Simulate two samples since PCA expects >1 row
#     features1 = features1.reshape(1, -1) if features1.ndim == 1 else features1
#     features2 = features2.reshape(1, -1) if features2.ndim == 1 else features2

#     # Concatenate features to simulate a multi-sample dataset for PCA
#     combined_features = np.vstack([features1, features2])
#     n_components = min(10, combined_features.shape[0], combined_features.shape[1])

#     if n_components > 1:  # Only apply PCA if reduction is possible
#         pca = PCA(n_components=n_components)
#         reduced_features = pca.fit_transform(combined_features)
#         features1, features2 = reduced_features[0], reduced_features[1]

#     # Calculate distances
#     # cosine_sim = 1 - cosine(features1, features2)
#     cosine_sim = compute_cosine_similarity(features1, features2)
#     euclidean_dist = euclidean(features1, features2)

#     print(f"Cosine similarity: {cosine_sim}")
#     print(f"Euclidean distance: {euclidean_dist}")

#     # Dynamic Time Warping
#     distance, _ = fastdtw(features1.flatten(), features2.flatten())
#     print(f"DTW distance: {distance}")

#     # Ensemble decision based on different distance metrics
#     similarity_score = (cosine_sim + (1 / (1 + euclidean_dist)) + (1 / (1 + distance))) / 3

#     print(f"Similarity score (ensemble): {similarity_score}")

#     # Decide based on threshold
#     return similarity_score > threshold




def compare_features(features1, features2, threshold=0.6):
    """
    Compare two feature vectors using multiple distance metrics.
    :param features1: Features of the first audio file.
    :param features2: Features of the second audio file.
    :param threshold: Similarity threshold to classify as same or different speaker.
    :return: True if both features indicate the same speaker, False otherwise.
    """
    # Normalize features to zero mean and unit variance
    features1 = (features1 - np.mean(features1)) / np.std(features1)
    features2 = (features2 - np.mean(features2)) / np.std(features2)

    # Calculate Euclidean Distance
    euclidean_dist = euclidean(features1, features2)

    # Calculate Manhattan Distance
    manhattan_dist = cityblock(features1, features2)

    # Calculate Correlation Coefficient
    corr_coeff, _ = pearsonr(features1.flatten(), features2.flatten())
    corr_coeff = max(-1, min(1, corr_coeff))  # Ensure it remains in [-1, 1]

    # Calculate Dynamic Time Warping
    dtw_distance, _ = fastdtw(features1.flatten(), features2.flatten())

    print(f"Euclidean distance: {euclidean_dist}")
    print(f"Manhattan distance: {manhattan_dist}")
    print(f"Correlation coefficient: {corr_coeff}")
    print(f"DTW distance: {dtw_distance}")

    # Compute Ensemble Similarity Score
    similarity_score = (
        (1 / (1 + euclidean_dist)) +    # Inverse of Euclidean Distance
        (1 / (1 + manhattan_dist)) +    # Inverse of Manhattan Distance
        (corr_coeff + 1) / 2 +          # Scale correlation coefficient to [0, 1]
        (1 / (1 + dtw_distance))        # Inverse of DTW distance
    ) / 4  # Average the metrics

    print(f"Similarity score (ensemble): {similarity_score}")

    # Decide based on threshold
    return similarity_score > threshold


def similarity_handler(file1, file2):

    # Extract features from both audio files
    features1 = extract_features(file1)
    features2 = extract_features(file2)

    # Compare the features
    result = compare_features(features1, features2)

    if result:

        print(f"The files {file1} and {file2} are from the same speaker.")
        return True
    else:
        print(f"The files {file1} and {file2} are from different speakers.")
        return False
