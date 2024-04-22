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
from scipy.signal import hilbert
import scipy.stats as stats
import soundfile as sf


# Path to the folder containing the txt files to be evaluated
#audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Testing_set'


audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Subset_features'

# audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\harmonics'
# metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")


#audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Subset_features'

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

#file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'


 
#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
# save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\subset_calls\\test_spectral_centroid_statistics'
save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\subset_calls\\test_rms_'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)


# for i in range(len(parameters)):
    # Iterate over each audio file
for file in tqdm(list_files):

#     onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
#     offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))
    
#     chick = os.path.basename(file)[:-4]

    ##### 1- Load audio file
    audio_y, sr = lb.load(file, sr=44100)
     
    duration = len(audio_y) / sr

    ##### 2- Apply the Bandpass filter
    # The Bandpass filter is applied to the audio signal to remove 
    # the background noise and keep only the chick's vocalizations ( among the frequencies 2000-12500 Hz)

    audio_fy = ut.bp_filter(audio_y, sr, lowcut=1600, highcut= 12700)

    # # show the waveform 
    # plt.figure(figsize=(30, 8))
    # plt.plot(audio_y)
    # plt.show()

    # Parameters for spectrogram and pitch estimation
    frame_length = 2048
    hop_length = 512
    win_length = frame_length // 2
    n_fft = 2048*2

    # Parameters for PYIN
    threshold = 100
    beta_parameters = (0.10, 0.10)
    fmin = 1800
    fmax = 12500
    resolution = 0.01

    # ## 3- Estimate pitch using PYIN    
    # f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_fy, sr=sr, frame_length=frame_length, hop_length=hop_length, 
    # fmin=fmin, fmax=fmax, n_thresholds=threshold, beta_parameters=beta_parameters, resolution=resolution)  # f0 is in hertz!!

    # # compute spectrogram
    # S = np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))

    # # segment spectrogram into calls

    # calls_S = ut.segment_spectrogram(S, onsets, offsets, sr=sr)

    
    # # f0_in_times = lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
    # # #save f0_pyin_lb to csv
    # # f0_pyin_lb_df = pd.DataFrame(f0_pyin_lb)
    # # # replace nan in dataframe with 0
    # # f0_pyin_lb_df = f0_pyin_lb_df.fillna(0)
    # # # save f0_pyin_lb to csv in frame 
    # # # Save pitch data to CSV
    # # f0_pyin_df = pd.DataFrame(f0_in_times, columns=['F0_in_Time'])
    # # f0_pyin_df.to_csv(chick + '_f0_pyin_.csv', index=False)
    # # #Extract calls
    # f0_calls = ut.get_calls_F0(f0_pyin_lb, onsets, offsets, sr, hop_length, n_fft)
    # call_numbers = []
    # F0_means = []
    # F0_stds = []
    # F0_skewnesses = []
    # F0_kurtosises = []
    # F1_means = []
    # F2_means = []
    # F0_F1_ratios = []
    # F0_F2_ratios = []
    # F0_fst_order_diffs = []


    # #### Compute for each calls statistics over the F0: Mean, Standard Deviation, Skewness, Kurtosis 
    # for i,  f0_call in enumerate(f0_calls):

    #     f0_call_without_nans = f0_call[~np.isnan(f0_call)]
    #     f0_call_nan_zeros = np.nan_to_num(f0_call)
    #     # compute the statistics
    #     f0_call_mean = f0_call_without_nans.mean()
    #     f0_call_std = f0_call_without_nans.std()
    #     f0_call_skewness = stats.skew(f0_call_without_nans)
    #     f0_call_kurtosis = stats.kurtosis(f0_call)
        
    #     call_numbers.append(i)
    #     F0_means.append(f0_call_mean)
    #     F0_stds.append(f0_call_std)
    #     F0_skewnesses.append(f0_call_skewness)
    #     F0_kurtosises.append(f0_call_kurtosis)

    #     # compute the 1st derivative of the F0
    #     f0_fst_order_diff = np.diff(f0_call)
    #     F0_fst_order_diffs.append(f0_fst_order_diff)


    #     F1_Hz = f0_call_nan_zeros*2    
    #     F2_Hz = f0_call_nan_zeros*3
    #     F1_Hz_withoutNans = f0_call_without_nans*2
    #     F2_Hz_withoutNans = f0_call_without_nans*3

    #     F1_Hz_mean = np.mean(F1_Hz_withoutNans)
    #     F2_Hz_mean = np.mean(F2_Hz_withoutNans)
    #     F1_means.append(F1_Hz_mean)
    #     F2_means.append(F2_Hz_mean)


    #     f0_frqbin = f0_call_mean * n_fft / sr
    #     f1_frqbin = F1_Hz_mean * n_fft / sr
    #     f2_frqbin = F2_Hz_mean * n_fft / sr

    #             # get magnitude at F1, and F2 and F0
    #     F0_mag =calls_S[i][int(f0_frqbin)]
    #     F1_mag =calls_S[i][int(f1_frqbin)]
    #     F2_mag =calls_S[i][int(f2_frqbin)]

    #     # compute magnitude ratios betwween f0 and f1, and f0 and f2
    #     F0_F1_ratio = F0_mag / F1_mag
    #     F0_F2_ratio = F0_mag / F2_mag
    #     F0_F1_ratios.append(F0_F1_ratio)
    #     F0_F2_ratios.append(F0_F2_ratio)

       
    #            #  visualise F0, F1 and F2 on top of the spectrogram
    # ####

    #     # Plot the linear spectrogram
    #     plt.figure(figsize=(10, 5))
    #     lb.display.specshow(lb.amplitude_to_db(calls_S[i], ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    #     times = lb.times_like(f0_call_nan_zeros, sr=sr, hop_length=hop_length)

    #     plt.plot(times, f0_call_nan_zeros, label='Pitch (Hz)', color='red')
    #     # plt.plot(f0_pyin_lb_nan_zeros, label='Pitch (Hz)', color='red')

    #     plt.plot(times, F1_Hz, label='F1 (Hz)', color='blue')
    #     plt.plot(times, F2_Hz, label='F2 (Hz)', color='green')
    #     # plt.colorbar(format='%+2.0f dB')
    #     plt.title('Linear Spectrogram and F0, F1, F2')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Frequency')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    

        
    # f0_features_calls = pd.DataFrame()
    # f0_features_calls['Call Number'] = call_numbers
    # f0_features_calls['F0 Mean'] = F0_means
    # f0_features_calls['F0 Std'] = F0_stds
    # f0_features_calls['F0 Skewness'] = F0_skewnesses
    # f0_features_calls['F0 Kurtosis'] = F0_kurtosises

    # f0_features_calls['F0 1st Order Diff'] = F0_fst_order_diffs
    # f0_features_calls['F1 Mean'] = F1_means
    # f0_features_calls['F2 Mean'] = F2_means
    # f0_features_calls['F0-F1 Ratio'] = F0_F1_ratios
    # f0_features_calls['F0-F2 Ratio'] = F0_F2_ratios


    # f0_features_calls.to_csv(os.path.join(save_results_folder, f"{chick}_F0_features.csv"), index=False)




















########################################################################################################################################

    
    # # Compute the analytic signal
    # analytic_signal = hilbert(audio_y)

    # # Compute the envelope
    # envelope = np.abs(analytic_signal)

    # # compute the peaks of the envelope # REVIEW THE DISTANCE
    # peaks, _ = signal.find_peaks(envelope, distance=1000)
    # # show the envelope
    # plt.figure(figsize=(30, 8))
    # plt.plot(envelope)
    # plt.show()
    # #save image
    # plt.savefig(os.path.join(save_results_folder, f"{chick}_envelope.png"))
    # # Compute the instantaneous phase
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))


















    #######################################################################################################################################
    # Compute the spectral bandwidth and then extract the mean
    lin_spec= np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))
    # Compute the frequencies
    frequencies = lb.fft_frequencies(sr=sr, n_fft=frame_length)

    energy_y = np.array([sum(abs(audio_fy[i:i+frame_length]**2))for i in range(0, len(audio_fy), hop_length)])

   
 ########################################################################################################################################
    
    # # #### 4- Segment the calls in wave files
    # calls_wave_file = ut.get_calls_waveform(audio_fy, onsets, offsets, sr= 44100)
    # for call in calls_wave_file:

        #Compute the rms (loudness) of the audio
        
        # rms_y = lb.feature.rms(y= audio_fy, frame_length=frame_length, hop_length=hop_length)
        
        # compute the mean and st.dev of the rms
        # mean_rms, st_dev_rms = np.mean(rms_call), np.std(rms_call)
        # # create dictionary with the mean and skewness of the rms
        # rms_call_statistics = {
        #     'Mean': mean_rms,
        #     'St.dev.': st_dev_rms
        # }

        # print(f'The main statistic of the call are', rms_call_statistics)
    # #Plot the linear spectrogram
    # plt.figure(figsize=(13, 8))
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # rms_y_times = lb.times_like(rms_y, sr=sr, hop_length=hop_length)
    # ax[0].semilogy(rms_y_times, rms_y[0], label='RMS Energy')
    # ax[0].set(xticks=[])
    # ax[0].legend()
    # ax[0].label_outer()
    # lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    # ax[1].set(xlabel='Time (s)', ylabel='Frequency')
    # # save the plot in the results folder
    # plt.savefig(os.path.join(save_results_folder, f"{file}_rms_energy.png"))

        #### 6- Compute the Envelope of the calls 
        # Compute the analytic signal

    analytic_signal = hilbert(audio_fy)
    # Compute the envelope
    envelope = np.abs(analytic_signal)

    #Plot the linear spectrogram
    fig, ax = plt.subplots(nrows=2, sharex=True)
    envelope_times = lb.times_like(envelope, sr=sr, hop_length=hop_length)
    ax[0].semilogy(envelope_times, envelope, label='Envelope')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    ax[1].set(xlabel='Time (s)', ylabel='Frequency')
    

    # Compute the instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # Compute the instantaneous frequency
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr
    # Compute the instantaneous amplitude
    instantaneous_amplitude = np.abs(analytic_signal)

    envelope_statistics = {
        'Envelope': envelope,
        'Instantaneous Phase': instantaneous_phase,
        'Instantaneous Frequency': instantaneous_frequency,
        'Instantaneous Amplitude': instantaneous_amplitude
    }

    print(f'The main statistic of the call are', instantaneous_frequency)
    print(f'The main statistic of the call are', instantaneous_amplitude)
    print(f'The main statistic of the call are', instantaneous_phase)
    print(f'The main statistic of the call are', envelope)
    print('work completed')


        # save the statistics to a json file
        # envelope_statistics_filename = os.path.join(save_results_folder, f"{chick}_envelope_statistics.json")
        # with open(envelope_statistics_filename, 'w') as file:    
        #     json.dump(envelope_statistics, file)

        
       
    ########################################################################################################################################
    
    
    
    #S, phase = lb.magphase(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))

    # Spectral_Centroid_mean = [] 
    # call_numbers = []
    # # Compute the spectrogram of the entire audio file
    # lin_spec= np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))
        
    # # Extract the spectrogram calls from the audio file
    # calls_s_files = ut.segment_spectrogram(spectrogram= S, onsets=onsets, offsets=offsets, sr=sr)
      

    # for i, call_s in enumerate(calls_s_files):
    #     call_numbers.append(i)
    #     # Compute the spectral centroid and then extract the mean
    # spectral_centroid_call = lb.feature.spectral_centroid(S= lin_spec, sr=sr, n_fft=frame_length, hop_length=hop_length)
    #     # deal with nans
    #     spectral_centroid_without_nans = spectral_centroid_call[~np.isnan(spectral_centroid_call)]
    #     # replace nans with zeros
    #     if spectral_centroid_without_nans.size == 0:
    #         Spectral_Centroid_mean.append(np.nan)
    #         continue 
    #     else:     
    #     # compute the mean of the spectral centroid
    #         mean_spectral_centroid = np.mean(spectral_centroid_without_nans)
    #         Spectral_Centroid_mean.append(mean_spectral_centroid)
    
    # spectral_centroid_feature_calls = pd.DataFrame()
    # spectral_centroid_feature_calls['Spectral Centroid Mean'] = mean_spectral_centroid

    # spectral_centroid_feature_calls.to_csv(os.path.join(save_results_folder, f"{chick}_spectral_centroid_features.csv"), index=False)

    # #Plot the linear spectrogram
    # plt.figure(figsize=(30, 8))
    # lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    # times = lb.times_like(spectral_centroid_call[0], sr=sr, hop_length=hop_length)

    # plt.plot(times, spectral_centroid_call[0], label='Spectral Centroid', color='green')
    # # plt.colorbar(format='%+2.0f dB')
    # plt.title('Linear Spectrogram and Spectral Centroid')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()
