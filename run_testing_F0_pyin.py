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


audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Sub_testing_set'


#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")


#audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Subset_features'

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

#file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'


 
#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\subset_calls\\test_spectral_centroid_statistics'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)



# for i in range(len(parameters)):
    # Iterate over each audio file
for file in tqdm(list_files):
    onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))
    
    chick = os.path.basename(file)[:-4]

    ##### 1- Load audio file
    audio_y, sr = lb.load(file, sr=44100)
     
    duration = len(audio_y) / sr

    ##### 2- Apply the Bandpass filter
    # The Bandpass filter is applied to the audio signal to remove 
    # the background noise and keep only the chick's vocalizations ( among the frequencies 2000-12500 Hz)

    audio_fy = ut.bp_filter(audio_y, sr, lowcut=1600, highcut=12700)

    # # show the waveform 
    # plt.figure(figsize=(30, 8))
    # plt.plot(audio_y)
    # plt.show()

    # Parameters for spectrogram and pitch estimation
    frame_length = 2048
    hop_length = 512
    win_length = frame_length // 2

    # Parameters for PYIN
    threshold = 100
    beta_parameters = (0.10, 0.10)
    fmin = 1800
    fmax = 12500
    resolution = 0.05   

    # ## 3- Estimate pitch using PYIN    
    # f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_y, sr=sr, frame_length=frame_length, hop_length=hop_length, 
    # fmin=fmin, fmax=fmax, n_thresholds=threshold, beta_parameters=beta_parameters, resolution=resolution)
    
    # f0_in_times = lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
    # #save f0_pyin_lb to csv
    # f0_pyin_lb_df = pd.DataFrame(f0_pyin_lb)
    # # replace nan in dataframe with 0
    # f0_pyin_lb_df = f0_pyin_lb_df.fillna(0)
    # # save f0_pyin_lb to csv in frame 
    # # Save pitch data to CSV
    # f0_pyin_df = pd.DataFrame(f0_in_times, columns=['F0_in_Time'])
    # f0_pyin_df.to_csv(chick + '_f0_pyin_.csv', index=False)
    # Extract calls
    # f0_calls = ut.get_calls_F0(f0_pyin_lb, onsets, offsets)
    # call_F0_statistics = []
    # #### Compute for each calls statistics over the F0: Mean, Standard Deviation, Skewness, Kurtosis 
    # for call in f0_calls:
    #     # convert call to a DataFrame
    #     f0_call = pd.DataFrame(call, columns=['F0'])
    #     # compute the statistics
    #     f0_call_mean, f0_call_std, f0_call_skewness, f0_call_kurtosis = f0_call['F0'].mean(), f0_call['F0'].std(), stats.skew(f0_call['F0']), stats.kurtosis(f0_call['F0'])
    #     # Append statistics to the list
    #     call_F0_statistics.append({
    #         #'Call Number': i,
    #         'Mean': f0_call_mean,
    #         'Standard Deviation': f0_call_std,
    #         'Skewness': f0_call_skewness,
    #         'Kurtosis': f0_call_kurtosis
    #     })
    # print(f'The main statistic of the call are', call_F0_statistics)

    # print(f'The main statistic of the call are', call_F0_statistics)
    # # save the statistics to a json file
    # f0_call_statistics_filename = os.path.join(save_results_folder, f"{chick}_F0_statistics.json")
    # with open(f0_call_statistics_filename, 'w') as file:
    #     json.dump(call_F0_statistics, file)

    ########################################################################################################################################

    # # Mapping the Hz to frequency bins


    # f0_f1_ratio = mean_f0 / mean_f1
    # f0_f2_ratio = mean_f0 / mean_f2

    # # Create a dictionary with the ratios
    # ratios = {
    #     'F0/F1': f0_f1_ratio,
    #     'F0/F2': f0_f2_ratio
    # }

    # # Save the ratios to a json file
    # ratios_filename = os.path.join(save_results_folder, f"{chick}_ratios.json")
    # with open(ratios_filename, 'w') as file:
    #     json.dump(ratios, file)

    # print(f'The main statistic of the call are', ratios)
    


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
    # Compute the spectral bandwidth and then extract the mean
    # lin_spec= np.abs(lb.stft(y=audio_y, n_fft=frame_length, hop_length=hop_length))
    # # Compute the frequencies
    # frequencies = lb.fft_frequencies(sr=sr, n_fft=frame_length)

    # energy_y = np.array([sum(abs(audio_y[i:i+frame_length]**2))for i in range(0, len(audio_y), hop_length)])

    #     # take the onsets and offsets from the metadata file
    # # # get ground truth (onsets, offsets)
    # onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    # offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt')) 

    
    # #### 4- Segment the calls in wave files
    # calls_wave_file = ut.get_calls_waveform(audio_fy, onsets, offsets, sr= 44100)
    # for call in calls_wave_file:

        # Compute the rms (loudness) of the audio
        # rms_call = lb.feature.rms(y=call, frame_length=frame_length, hop_length=hop_length)
        
        # # compute the mean and st.dev of the rms
        # mean_rms, st_dev_rms = np.mean(rms_call), np.std(rms_call)
        # # create dictionary with the mean and skewness of the rms
        # rms_call_statistics = {
        #     'Mean': mean_rms,
        #     'St.dev.': st_dev_rms
        # }

    #     print(f'The main statistic of the call are', rms_call_statistics)


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
    # # Save the plot with the chick's name
    # plt.savefig(os.path.join(save_results_folder, f"{chick}_rms_.png"))

        #### 6- Compute the Envelope of the calls 
        # Compute the analytic signal
        # analytic_signal = hilbert(call)
        # # Compute the envelope

        # envelope = np.abs(analytic_signal)
        # # Compute the instantaneous phase
        # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        # # Compute the instantaneous frequency
        # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr
        # # Compute the instantaneous amplitude
        # instantaneous_amplitude = np.abs(analytic_signal)

        # envelope_statistics = {
        #     'Envelope': envelope,
        #     'Instantaneous Phase': instantaneous_phase,
        #     'Instantaneous Frequency': instantaneous_frequency,
        #     'Instantaneous Amplitude': instantaneous_amplitude
        # }

        # print(f'The main statistic of the call are', instantaneous_frequency)
        # print(f'The main statistic of the call are', instantaneous_amplitude)
        # print(f'The main statistic of the call are', instantaneous_phase)
        # print(f'The main statistic of the call are', envelope)
        # print('work completed')


        # save the statistics to a json file
        # envelope_statistics_filename = os.path.join(save_results_folder, f"{chick}_envelope_statistics.json")
        # with open(envelope_statistics_filename, 'w') as file:    
        #     json.dump(envelope_statistics, file)

        
       
    ########################################################################################################################################
    S, phase = lb.magphase(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))
    # Compute the spectrogram of the entire audio file
    # spectrogram = np.abs(lb.stft(y=audio_y, n_fft=frame_length, hop_length=hop_length))
    spectral_centroid_mean = []
    # Compute spectrogram of single call
    calls_s_files = ut.segment_spectrogram(spectrogram= S, onsets=onsets, offsets=offsets, sr=sr)
        #### 8- Compute the mean of the Mean of the Spectral centroid
    for call_s in calls_s_files:


        # test spectral segmenttion


        # Compute the spectral centroid and then extract the mean
        spectral_centroid_call = lb.feature.spectral_centroid(S=call_s, sr=sr, n_fft=frame_length, hop_length=hop_length)
        # spectral_centroid_call= ut.spectral_centroid(S=call_s, sr=sr, n_fft=frame_length, hop_length=hop_length)

        # # compute the times for the spectral centroid
        # spectral_centroid_times = lb.times_like(spectral_centroid[0], sr=sr, hop_length=hop_length)
        # # save the spectral centroid to csv
        # spectral_centroid_df = pd.DataFrame(spectral_centroid, columns=['Spectral_Centroid'])

        mean_spectral_centroid = np.mean(spectral_centroid_call)
        # create dictionary with the mean of the spectral centroid
        

        # create dictionary with the mean of the spectral centroid
        spectral_centroid_mean.append(mean_spectral_centroid)
        print(f'The mean spectral centroid of the call is', mean_spectral_centroid)


        # save the spectral centroid statistics to a json file
    spectral_centroid_statistics_filename = os.path.join(save_results_folder, f"{chick}_spectral_centroid_statistics.json")
    with open(spectral_centroid_statistics_filename, 'w') as file:
        json.dump(spectral_centroid_mean, file)

    # # Compute the mean of the Mean of the Spectral centroid   









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


    # # # Compute Mel spectrogram
    # S = lb.feature.melspectrogram(y=audio_y, sr=sr, win_length=win_length, hop_length=hop_length, n_mels=128)
    # S_db = lb.amplitude_to_db(S, ref=np.max)
    # # Plot Mel spectrogram
    # plt.figure(figsize=(30, 8))
    # lb.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', alpha=0.75)
    
    # # Plot pitch (F0)
    # plt.plot(f0_in_times, f0_pyin_lb, label='Pitch (Hz)', color='red')

    # # # Plot the RMS
    # # plt.plot(rms_y_times, rms_y_transposed, label='RMS', color='white')

    # # Plot the spectral centroid
    # #plt.plot(spectral_centroid_times[0], spectral_centroid[0], label='Spectral Centroid', color='green')
    # #plt.plot(spectral_centroid_times, spectral_centroid[0], label='Spectral Centroid', color='green')


    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectral Centroid in Mel-spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.tight_layout()
    
    # # # Save the plot with the chick's name
    # plt.savefig(os.path.join(save_results_folder, f"{chick}_F0_thr_{threshold}_beta_{beta_parameters}_fmin_{fmin}_fmax_{fmax}_resolution_{resolution}.png"))
    
    # Show the plot
    #plt.show()