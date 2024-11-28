from speaker_verification_eda import get_path_to_wave
import random
import os

def compute_verification_accuracy(base_path="/Users/bikrant-bikram/Downloads/wav", iterations=100):

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
