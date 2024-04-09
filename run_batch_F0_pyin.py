import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import evaluation as my_eval
import utils as ut
import basic_pitch as bp
import matplotlib.pyplot as plt
import librosa as lb
import scipy.signal as signal
from scipy.signal import hilbert
import scipy.stats as stats
import soundfile as sf

# create dictionary with the duration of the calls for each chick

# Path to the folder containing the txt files to be evaluated
audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'

metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\testing_calls\\test_batch_F0_pyin'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))
#file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'
call_statistics = []

for file in tqdm(list_files):
   
    chick = os.path.basename(file)[:-4]


    # Create a folder with the chick's name
    chick_folder = os.path.join(save_results_folder, chick)
    if not os.path.exists(chick_folder):
        os.makedirs(chick_folder)
    
    # take the onsets and offsets from the metadata file
    # # get ground truth (onsets, offsets)
    onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt')) 


    ##### 1- Load audio file
    audio_y, sr = lb.load(file, sr=44100, duration=10.0)



    ##### 2- Apply the Bandpass filter
    # The Bandpass filter is applied to the audio signal to remove 
    # the background noise and keep only the chick's vocalizations (that usually range among the frequencies 2000-12000 Hz)

    audio_y = ut.bp_filter(audio_y, sr, lowcut=1800, highcut=12500)

    # Parameters for spectrogram and pitch estimation
    frame_length = 2048
    hop_length = 512
    win_length = frame_length // 2

    # Parameters for PYIN
    threshold = 100
    beta_parameters = (0.10, 0.10)
    fmin = 1800
    fmax = 12500
    resolution = 0.01   



   #### 3- Estimate pitch using PYIN  
     
    f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_y, sr=sr, frame_length=frame_length, hop_length=hop_length, 
    fmin=fmin, fmax=fmax, n_thresholds=threshold, beta_parameters=beta_parameters, resolution=resolution)
    
    f0_in_times = lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
    #save f0_pyin_lb to csv
    f0_pyin_lb_df = pd.DataFrame(f0_pyin_lb)
    # replace nan in dataframe with 0
    f0_pyin_lb_df = f0_pyin_lb_df.fillna(0)
   
    # Extract calls
    f0_calls = ut.get_calls_F0(f0_in_times, f0_pyin_lb, onsets, offsets)

    #### Compute for each calls statistics over the F0: Mean, Standard Deviation, Skewness, Kurtosis 
    for i, call in enumerate(f0_calls):
        # convert call to a DataFrame
        f0_call = pd.DataFrame(call, columns=['F0'])
        # Calculate statistics for the call segment
        f0_call_mean = np.mean(f0_call)
        f0_call_std = np.std(f0_call)
        f0_call_skewness = f0_call.skew()
        f0_call_kurtosis = f0_call.kurtosis()
        
        # Append statistics to the list
        call_statistics.append({
            'Call Number': i,
            'Mean': f0_call_mean,
            'Standard Deviation': f0_call_std,
            'Skewness': f0_call_skewness,
            'Kurtosis': f0_call_kurtosis
        })

    # save all the statistics of all the calls in a csv file

    # Convert the list of dictionaries to a DataFrame
    call_statistics_df = pd.DataFrame(call_statistics)

    F0_statistics_filename = os.path.join(chick_folder, f"{chick}_F0_statistics.csv")
    call_statistics_df.to_csv(F0_statistics_filename, index=False)





    #### 4- Segment the calls in wave files

    calls_wave_file = ut.segment_calls(audio_y, sr, onsets, offsets, chick_folder)





    #### 5- Compute the RMS of each call                                                                         
    for call_wave_file in calls_wave_file:
        # compute the rms of the call
        rms_y = lb.feature.rms(y = call_wave_file, frame_length=frame_length, hop_length=hop_length)
        # compute the times for the rm
        audio_duration = len(audio_y) / sr

        rms_y_times = lb.frames_to_time(np.arange(rms_y.shape[-1]), sr=sr, hop_length=hop_length)
        # save the rms of the call
        rms_y_df = pd.DataFrame(rms_y, columns=['RMS'])
        # save the rms of the call to csv
        rms_y_df.to_csv(chick + '_rms.csv', index=False)






    #### 6- Compute the Envelope of the calls 

        # Compute the analytic signal
        analytic_signal = hilbert(audio_y)

        # Compute the envelope
        envelope = np.abs(analytic_signal)





    #### 7- Compute the spectrogram of the entire audio file

    lin_spec= np.abs(lb.stft(y=audio_y, n_fft=frame_length, hop_length=hop_length))

    # Compute spectrogram of single call
    calls_spec_files = ut.segment_calls(lin_spec, sr, onsets, offsets, chick_folder)

    #### 8- Compute the mean of the Spectral centroid
    for call_spec_file in calls_spec_files:
        # Compute the spectral centroid and then extract the mean
        spectral_centroid = lb.feature.spectral_centroid(S=call_spec_file, sr=sr, n_fft=frame_length, hop_length=hop_length)
        # compute the times for the spectral centroid
        spectral_centroid_times = lb.times_like(spectral_centroid[0], sr=sr, hop_length=hop_length)
        # save the spectral centroid to csv
        spectral_centroid_df = pd.DataFrame(spectral_centroid, columns=['Spectral_Centroid'])
        spectral_centroid_df.to_csv(chick + '_spectral_centroid.csv', index=False)
