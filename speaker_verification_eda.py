

import os
import pandas as pd
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# print("\033[93m","==========================[ Metadata ]============================")

# Function takes the path of the wave file and return the duration of the gilr in seconds
def get_duration(file_path: str):
   audio_data, sample_rate = librosa.load(file_path)
   duration = librosa.get_duration(y=audio_data, sr=sample_rate)
   return duration

# function takes the path to the main Directory of DataSet and return the MetaDeta of the dataSet in the form of the dataframe
def get_path_to_wave( base_path="/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification")-> pd.DataFrame:

    # df.columns = [["fileName", "userId", "Duration", "Path"]]
    fileName = []
    userId = []
    duration = []
    path = []

    files_at_root = os.listdir(base_path)

    for id in files_at_root:
        # print(id)
        if(os.path.isdir(base_path+ "/"+ id) == False):
            continue
        files_under_id=os.listdir(base_path+ "/"+ id)

        for sub_folders in files_under_id:
            # print(sub_folders)
            if(os.path.isdir(base_path+ "/"+ id+"/"+sub_folders) == False):
                continue
            wave_files_in_sub_folder =os.listdir( base_path+ "/"+ id+"/"+sub_folders)
            for wave_file in wave_files_in_sub_folder:
                total_path = base_path+ "/"+ id+"/"+sub_folders+"/"+wave_file
                fileName.append(wave_file)
                userId.append(id)
                duration.append(get_duration(total_path))
                path.append(total_path)
                # new_row = {"fileName" : wave_file, "userId" :  id , "Duration" : 0 , "Path": total_path}
                # df._append(new_row, ignore_index=True)
    df = pd.DataFrame({"fileName" : fileName, "userId" :  userId , "Duration" : duration , "Path": path})
    return df


def metadata_visualizer():

    metadata = get_path_to_wave()
    ids = metadata["userId"].unique()
    print("Total ID:  " ,len(ids) )
    print("IDs  are : ", " ".join(ids))

    print()
    print("Audio File Description.")
    print(metadata["Duration"].describe())
    print("Dataset contains total of ", sum(metadata["Duration"]), "Seconds of files...")


    print()
    print("IDs and there respective total Duration of audio files")
    print(metadata.groupby ("userId", as_index=False) ["Duration"].sum())

    print()
    print("IDs and there respective total no of audio files")
    print(metadata.groupby ("userId", as_index=False) ["Path"].count())
    return metadata


# Takes path of wave file output directeto and other argumnets and makes chunk for that perticular file if needed
def audio_to_chunks(file_path, output_dir, chunk_length, overlap, fileName, until = 83841):

    audio, sr = librosa.load(file_path, sr=None)
    chunk_samples = chunk_length * sr
    overlap_samples = overlap * sr
    chunks = []
    start = 0
    while start < len(audio):
        end = start + chunk_samples
        # if(end - start-chunk_length < 0):
        #     break
        chunk = audio[start:end]
        if len(chunk) >= until:
            # print(len(chunk), start, end)
            chunks.append(chunk)
        else:
            break
        start += (chunk_samples - overlap_samples)  # move by chunk length minus overlap
    for i, chunk in enumerate(chunks):
        chunk_file_path = f'{output_dir}/{fileName}_chunk_{i+1}.wav'
        print(chunk_file_path)
        sf.write(chunk_file_path, chunk, sr)


# This Function takes dataframe of metadata and makes chunks from that information
def make_chunks(metadata,base_path = "/Users/bikrant-bikram/Coding/Projects/SpeakerVerification/SpeakerVerification/"):
    try:
        for i in metadata.loc():
            fileName,userId, Duration , path  = i["fileName"],i["userId"],  i["Duration"], i["Path"]
            print(i)
            t= path
            output_dir = base_path +userId +"/"+ t.split("/")[-2]
            audio_to_chunks(path, output_dir, 8, 7, fileName)
    except:
        print("ERROR")


# metadata = metadata_visualizer()
# print(metadata)
# # make_chunks(metadata)
