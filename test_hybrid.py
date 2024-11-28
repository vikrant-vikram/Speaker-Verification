
from speaker_verification_eda import get_path_to_wave



import pandas as pd
import random
from hybrid import handler
import librosa
import numpy as np

from hybrid import extract_svm_features_segments
import os

import gmm

def load_and_validate_audio(file_path):

    try:
        if extract_svm_features_segments(file_path) == None:
            return None
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None

        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Check if audio is valid (no NaN or Inf values)
        if not np.all(np.isfinite(y)):
            print(f"Error: Audio file {file_path} contains invalid data (NaN or Inf).")
            return None

        # Check if the audio is too short (e.g., empty audio)
        if len(y) == 0:
            print(f"Error: Audio file {file_path} is empty.")
            return None

        # Check if the sample rate is valid
        if sr <= 0:
            print(f"Error: Invalid sample rate for {file_path}.")
            return None

        return y, sr  # Return valid audio data and sample rate

    except librosa.util.exceptions.ParameterError as e:
        # Catch Librosa-specific errors (e.g., if the file is not a valid audio file)
        print(f"Librosa error processing file {file_path}: {e}")
        return None
    except FileNotFoundError as e:
        # Catch errors related to missing files
        print(f"File not found: {file_path}. Error: {e}")
        return None
    except ValueError as e:
        # Catch errors related to incorrect value types (e.g., in file format)
        print(f"ValueError: Issue with file {file_path}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error processing file {file_path}: {e}")
        return None

def extract_features_with_validation(file_path, n_mfcc=13):
    """
    Extract features from an audio file after validating the audio.
    """
    audio_data = load_and_validate_audio(file_path)



# def compute_verification_accuracy(base_path="/Users/bikrant-bikram/Downloads/wav", iterations=100):
#     """
#     Compute the accuracy of the speaker verification function over multiple iterations.

#     :param base_path: The base directory containing the audio files.
#     :param speaker_verification_function: Function that accepts two file paths
#            and returns True if they belong to the same speaker, otherwise False.
#     :param iterations: Number of random pairs to evaluate.
#     :return: Accuracy as a percentage and detailed results.
#     """
#     # Get the DataFrame with file paths and IDs
#     df = get_path_to_wave(base_path)

#     # Ensure there are enough files to compare
#     if len(df) < 2:
#         raise ValueError("Not enough files in the dataset to perform verification.")

#     # Initialize counters for correct predictions
#     correct_predictions = 0
#     detailed_results = []

#     # Run the loop for the specified number of iterations
#     for _ in range(iterations):
#         # Randomly select two audio files
#         file1, file2 = df.sample(n=2).to_dict(orient="records")

#         # Extract file paths and user IDs
#         path1, id1 = file1["Path"], file1["userId"]
#         path2, id2 = file2["Path"], file2["userId"]

#         # Ground truth: Are the two files from the same user?
#         ground_truth = (id1 == id2)

#         # Prediction from the speaker verification function
#         if load_and_validate_audio(path1) == None and load_and_validate_audio(path1) ==None:
#             continue



#         prediction = handler(path1, path2)

#         # Evaluate correctness
#         if(ground_truth == prediction):
#             print("corect =================================")
#             correct_predictions += 1
#         else:
#             print("===================================wrong")

#         print(id1, id2)

#         is_correct = (ground_truth == prediction)

#         # # Track the result
#         # correct_predictions += int(is_correct)
#         detailed_results.append({
#             "file1": file1,
#             "file2": file2,
#             "ground_truth_same_user": ground_truth,
#             "prediction": prediction,
#             "correct_prediction": is_correct
#         })

#     # Calculate accuracy
#     accuracy = (correct_predictions / iterations) * 100

#     return accuracy


# if __name__ == "__main__":
#     print(compute_verification_accuracy())


def compute_verification_accuracy(base_path="/Users/bikrant-bikram/Downloads/wav2", iterations=100):

    # Get the DataFrame with file paths and IDs
    df = get_path_to_wave(base_path)

    # Ensure there are enough files to create pairs
    if len(df) < 2:
        raise ValueError("Not enough files in the dataset to perform verification.")

    # Group files by userId
    grouped = df.groupby("userId")

    # Prepare containers for same-speaker and different-speaker pairs
    same_speaker_pairs = []
    different_speaker_pairs = []

    # Generate all same-speaker pairs
    for user_id, group in grouped:
        group_files = group.to_dict(orient="records")
        same_speaker_pairs.extend([(a, b) for i, a in enumerate(group_files) for b in group_files[i + 1:]])

    # Generate all possible different-speaker pairs
    all_files = df.to_dict(orient="records")
    while len(different_speaker_pairs) < len(same_speaker_pairs):
        file1, file2 = random.sample(all_files, 2)
        if file1["userId"] != file2["userId"]:
            different_speaker_pairs.append((file1, file2))

    random.shuffle(same_speaker_pairs)
    random.shuffle(different_speaker_pairs)
    half_iterations = iterations // 2
    selected_same_speaker_pairs = same_speaker_pairs[:half_iterations]
    selected_different_speaker_pairs = different_speaker_pairs[:half_iterations]
    selected_pairs = selected_same_speaker_pairs + selected_different_speaker_pairs
    random.shuffle(selected_pairs)
    correct_predictions = 0
    detailed_results = []
    for file1, file2 in selected_pairs:
        path1, id1 = file1["Path"], file1["userId"]
        path2, id2 = file2["Path"], file2["userId"]
        ground_truth = (id1 == id2)
        if load_and_validate_audio(path1) is None or load_and_validate_audio(path2) is None:
            continue





        prediction = handler(path1, path2)
        #


        # Check correctness
        is_correct = (ground_truth == prediction)
        print(id1, id2)
        if is_correct:
            print("==========================+CORRRECT ======================")

            correct_predictions += 1
        else:
            print("===========================WRONG==========================")
        # Track the result
        detailed_results.append({
            "file1": file1,
            "file2": file2,
            "ground_truth_same_user": ground_truth,
            "prediction": prediction,
            "correct_prediction": is_correct
        })

    # Calculate accuracy
    accuracy = (correct_predictions / iterations) * 100

    return accuracy, detailed_results


if __name__ == "__main__":
    accuracy, results = compute_verification_accuracy()
    print(f"Verification Accuracy: {accuracy}%")
