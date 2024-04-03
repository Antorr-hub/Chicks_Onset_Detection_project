import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import evaluation as my_eval
import json
import utils as ut
import basic_pitch as bp
import time
import matplotlib.pyplot as plt
import librosa as lb
import utils as ut

# create dictionary with the duration of the calls for each chick

# Path to the folder containing the txt files to be evaluated
audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Testing_set'

metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\testing_calls'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))
#file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'
call_statistics = []

for file in tqdm(list_files):
   
    chick = os.path.basename(file)[:-4]
    # take the onsets and offsets from the metadata file
    # # get ground truth (onsets, offsets)
    onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt')) 

    # Load audio file
    audio_y, sr = lb.load(file, sr=22050, duration=20.0)

    # Parameters for spectrogram and pitch estimation
    frame_length = 2048
    hop_length = 512
    n_fft = frame_length * 2
    win_length = frame_length // 2



    f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_y, sr=sr, frame_length=frame_length, hop_length=hop_length, fmin=2050, fmax=10000,
                        n_thresholds=100, beta_parameters = (0.15,0.15))
    

    # compute the rms of the audio
    rms_y = lb.feature.rms(y=audio_y, frame_length=frame_length, hop_length=hop_length)

    
    f0_in_times= lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
    
    # save f0_pyin_lb to csv
    f0_pyin_df = pd.DataFrame(f0_in_times, columns=['F0_in_Time'])
    # replace nan in dataframe with 0
    f0_pyin_df = f0_pyin_df.fillna(0)

    
    # Extract calls
    f0_calls = ut.get_calls_F0(f0_in_times, f0_pyin_lb, onsets, offsets)


    # Create a folder with the chick's name
    chick_folder = os.path.join(save_results_folder, chick)
    if not os.path.exists(chick_folder):
        os.makedirs(chick_folder)
    
    # Save each call segment to the chick's folder
    for i, call in enumerate(f0_calls):
        # convert call to a DataFrame
        f0_call = pd.DataFrame(call, columns=['F0'])
        call_filename = os.path.join(chick_folder, f"{chick}_f0_call_{i}.csv")
        call.to_csv(call_filename, index=False)


    # Iterate over each call file in the chick's folder
    for i in range(len(f0_calls)):
        call_filename = os.path.join(chick_folder, f"{chick}_f0_call_{i}.csv")
        # Load the call data from CSV into a DataFrame
        call_data = pd.read_csv(call_filename)
        
        # Calculate statistics for the call segment
        call_mean = np.mean(call_data['F0'])
        call_std = np.std(call_data['F0'])
        call_skewness = call_data['F0'].skew()
        call_kurtosis = call_data['F0'].kurtosis()
        
        # Append statistics to the list
        call_statistics.append({
            'Call Number': i,
            'Mean': call_mean,
            'Standard Deviation': call_std,
            'Skewness': call_skewness,
            'Kurtosis': call_kurtosis
        })

    # Convert the list of dictionaries to a DataFrame
    call_statistics_df = pd.DataFrame(call_statistics)

    # Save the statistics to a CSV file
    stats_filename = os.path.join(chick_folder, f"{chick}_call_statistics.csv")
    call_statistics_df.to_csv(stats_filename, index=False)



