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
import scipy.signal as signal
import scipy.stats as stats
import soundfile as sf


# Path to the folder containing the txt files to be evaluated
#audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Testing_set'

#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")


audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Subset_features'

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

#file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'



#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\subset_calls\\new_test'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)


# for i in range(len(parameters)):
    # Iterate over each audio file
for file in tqdm(list_files):
    
    chick = os.path.basename(file)[:-4]

    ##### 1- Load audio file
    audio_y, sr = lb.load(file, sr=44100)


    ##### 2- Apply the Bandpass filter
    # The Bandpass filter is applied to the audio signal to remove 
    # the background noise and keep only the chick's vocalizations ( among the frequencies 2000-12500 Hz)

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

    # # # Estimate pitch using PYIN    
    f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_y, sr=sr, frame_length=frame_length, hop_length=hop_length, 
    fmin=fmin, fmax=fmax, n_thresholds=threshold, beta_parameters=beta_parameters, resolution=resolution)
    
    f0_in_times = lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
    #save f0_pyin_lb to csv
    f0_pyin_lb_df = pd.DataFrame(f0_pyin_lb)
    # replace nan in dataframe with 0
    f0_pyin_lb_df = f0_pyin_lb_df.fillna(0)
    # save f0_pyin_lb to csv in frame 
    # Save pitch data to CSV
    f0_pyin_df = pd.DataFrame(f0_in_times, columns=['F0_in_Time'])
    f0_pyin_df.to_csv(chick + '_f0_pyin_.csv', index=False)


   
    ########################################################################################################################################

    # # Compute the first derivative of the frequency
    # f0_pyin_diff = np.diff(f0_pyin_lb)

    # # Compute the times for the first derivative of the frequency
    # f0_pyin_diff_times = f0_in_times[:-1]

    # # Plot the first derivative of the frequency
    # plt.figure(figsize=(10, 5))
    # plt.plot(f0_pyin_diff_times, f0_pyin_diff, label='First Derivative of Frequency', color='green')

   
    ########################################################################################################################################
    # Compute the spectral centroid and then extract the mean
    #spectral_centroid = lb.feature.spectral_centroid(y=audio_y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    # compute the times for the spectral centroid
    #spectral_centroid_times = lb.times_like(spectral_centroid[0], sr=sr, hop_length=hop_length)



    #######################################################################################################################################
    # # Compute the spectral bandwidth and then extract the mean
    # lin_spec= np.abs(lb.stft(y=audio_y, n_fft=frame_length, hop_length=hop_length))
    # # Compute the frequencies
    # frequencies = lb.fft_frequencies(sr=sr, n_fft=frame_length)

    # energy_y = np.array([sum(abs(audio_y[i:i+frame_length]**2))for i in range(0, len(audio_y), hop_length)])



    # # Apply the mask to the frequencies CREATE A  FUNCTION MASK FOR THE FREQUENCIES
    # low_mask = np.logical_and(frequencies >= 0, frequencies <= 2000)
    # lin_spec[low_mask] = 0
    # # Apply second mask over the frequencies between 8000 and 11025
    # high_mask = np.logical_and(frequencies >= 12500, frequencies <= 20000)
    # lin_spec[high_mask] = 0


    # # Compute the rms (loudness) of the audio
    # rms_y = lb.feature.rms(S=lin_spec, frame_length=frame_length, hop_length=hop_length)
    # # compute the times for the rm
    # audio_duration = len(audio_y) / sr

    # rms_y_times = lb.frames_to_time(np.arange(rms_y.shape[-1]), sr=sr, hop_length=hop_length)
    # # create a dataframe with the rms values
    # rms_y_df = pd.DataFrame(rms_y_times, columns=['RMS'])
    # # save rms_y to csv
    # rms_y_df.to_csv(chick + '_rms_.csv', index=False)
    # # Plot the linear spectrogram
    # plt.figure(figsize=(30, 8))
    # lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)

    # # Transpose if the matrix is not in the right shape
    # rms_y_transposed = rms_y.T
    # # Plot the RMS
    # plt.plot(rms_y_times, rms_y_transposed, label='RMS', color='white')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Linear Spectrogram and RMS')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.tight_layout()


    ########################################################################################################################################
    # plt.tight_layout()
    # Spec, phase = lb.magphase(lb.stft(y=audio_y))
    # rms_y= lb.feature.rms(y= Spec, frame_length=frame_length, hop_length=hop_length)
    # rms_y_times = lb.times_like(rms_y, sr=sr, hop_length=hop_length) 
    # # # save rms_y to csv
    # # rms_y_df = pd.DataFrame(rms_y)
    # # # replace nan in dataframe with 0
    # # rms_y_df = rms_y_df.fillna(0)
    # # # save rms_y to csv in frame
    # # # Save pitch data to CSV
    # # rms_y_df = pd.DataFrame(rms_y, columns=['RMS'])
    # # rms_y_df.to_csv(chick + '_rms_.csv', index=False)


    # # Compute Mel spectrogram
    S = lb.feature.melspectrogram(y=audio_y, sr=sr, win_length=win_length, hop_length=hop_length, n_mels=128)
    S_db = lb.amplitude_to_db(S, ref=np.max)
    # Plot Mel spectrogram
    plt.figure(figsize=(30, 8))
    lb.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', alpha=0.75)
    
    # Plot pitch (F0)
    #plt.plot(f0_in_times, f0_pyin_lb, label='Pitch (Hz)', color='red')

    # # Plot the RMS
    # plt.plot(rms_y_times, rms_y_transposed, label='RMS', color='white')

    # Plot the spectral centroid
    #plt.plot(spectral_centroid_times[0], spectral_centroid[0], label='Spectral Centroid', color='green')
    #plt.plot(spectral_centroid_times, spectral_centroid[0], label='Spectral Centroid', color='green')


    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectral Centroid in Mel-spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # # Save the plot with the chick's name
    plt.savefig(os.path.join(save_results_folder, f"{chick}_F0_thr_{threshold}_beta_{beta_parameters}_fmin_{fmin}_fmax_{fmax}_resolution_{resolution}.png"))
    
    # Show the plot
    #plt.show()