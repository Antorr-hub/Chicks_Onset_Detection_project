# Libraries used for the function of feature extraction: numpy, os, pandas, librosa, matplotlib, scipy, soundfile
import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import utils as ut
import scipy.signal as signal
from scipy.signal import hilbert
import scipy.stats as stats
import soundfile as sf
import evaluation as my_eval

frame_length = 2048
hop_length = 512
win_length = frame_length // 2
n_fft = 2048*2

def visualise_spectrogram_and_harmonics(spec, F0, F1, F2, sr, hop_length):
        # Plot the linear spectrogram
    plt.figure(figsize=(10, 5))
    lb.display.specshow(lb.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    times = lb.times_like(F0, sr=sr, hop_length=hop_length)

    plt.plot(times, F0, label='Pitch (Hz)', color='red')
    # plt.plot(f0_pyin_lb_nan_zeros, label='Pitch (Hz)', color='red')

    plt.plot(times, F1, label='F1 (Hz)', color='blue')
    plt.plot(times, F2, label='F2 (Hz)', color='green')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Linear Spectrogram and F0, F1, F2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return

def compute_f0_features(audio, features_data ,sr, hop_length, frame_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution):

    ''' run the pitch estimation using PYIN and compute the F0 features
    computes F1 and F2 based on F0 and computes the ratio of magnitudes between F0 and F1 and F0 and F2
    
    returns: return dataframe containing features for each call in recording
    
    '''
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec
    # ## 3- Estimate pitch using PYIN    
    f0_pyin_lb, _, _ = lb.pyin(audio, sr=sr, frame_length=frame_length, hop_length=hop_length, 
                               fmin=pyin_fmin_hz, fmax=pyin_fmax_hz, n_thresholds=pyin_ths, beta_parameters=pyin_beta,
                               resolution= pyin_resolution)  # f0 is in hertz!!

    # compute spectrogram
    S = np.abs(lb.stft(y=audio, n_fft=frame_length, hop_length=hop_length))

    # segment spectrogram into calls
    calls_S = ut.segment_spectrogram(S, onsets_sec, offsets_sec, sr=sr)

    # Compute for each calls statistics over the F0: Mean, Standard Deviation, Skewness, Kurtosis
    f0_calls = ut.get_calls_F0(f0_pyin_lb, onsets_sec, offsets_sec, sr, hop_length, n_fft)
    call_numbers = []
    F0_means = []
    F0_stds = []
    F0_skewnesses = []
    F0_kurtosises = []
    F1_means = []
    F2_means = []
    F0_F1_ratios = []
    F0_F2_ratios = []
    F0_fst_order_diffs = []


    # compute the statistics for each call 
    for i,  f0_call in enumerate(f0_calls):

        f0_call_without_nans = f0_call[~np.isnan(f0_call)]
        # compute the statistics
        f0_call_mean = f0_call_without_nans.mean()
        f0_call_std = f0_call_without_nans.std()
        f0_call_skewness = stats.skew(f0_call_without_nans)
        f0_call_kurtosis = stats.kurtosis(f0_call)
        
        call_numbers.append(i)
        F0_means.append(f0_call_mean)
        F0_stds.append(f0_call_std)
        F0_skewnesses.append(f0_call_skewness)
        F0_kurtosises.append(f0_call_kurtosis)

        # compute the 1st derivative of the F0
        f0_fst_order_diff = np.diff(f0_call)
        F0_fst_order_diffs.append(f0_fst_order_diff)

        F1_Hz_withoutNans = f0_call_without_nans*2
        F2_Hz_withoutNans = f0_call_without_nans*3

        F1_Hz_mean = np.mean(F1_Hz_withoutNans)
        F2_Hz_mean = np.mean(F2_Hz_withoutNans)
        F1_means.append(F1_Hz_mean)
        F2_means.append(F2_Hz_mean)


        f0_frqbin = f0_call_mean * n_fft / sr
        f1_frqbin = F1_Hz_mean * n_fft / sr
        f2_frqbin = F2_Hz_mean * n_fft / sr

                # get magnitude at F1, and F2 and F0
        F0_mag =calls_S[i][int(f0_frqbin)]
        F1_mag =calls_S[i][int(f1_frqbin)]
        F2_mag =calls_S[i][int(f2_frqbin)]

        # compute magnitude ratios betwween f0 and f1, and f0 and f2
        F0_F1_ratio = F0_mag / F1_mag
        F0_F2_ratio = F0_mag / F2_mag
        F0_F1_ratios.append(F0_F1_ratio)
        F0_F2_ratios.append(F0_F2_ratio)

       
    #  visualise F0, F1 and F2 on top of the spectrogram
    # visualise_spectrogram_and_harmonics(S, f0_pyin_lb, F1_Hz, F2_Hz, sr, hop_length)

    f0_features_calls = pd.DataFrame()
    f0_features_calls['Call Number'] = call_numbers
    f0_features_calls['F0 Mean'] = F0_means
    f0_features_calls['F0 Std'] = F0_stds
    f0_features_calls['F0 Skewness'] = F0_skewnesses
    f0_features_calls['F0 Kurtosis'] = F0_kurtosises

    f0_features_calls['F0 1st Order Diff'] = F0_fst_order_diffs
    f0_features_calls['F1 Mean'] = F1_means
    f0_features_calls['F2 Mean'] = F2_means
    f0_features_calls['F0-F1 Ratio'] = F0_F1_ratios
    f0_features_calls['F0-F2 Ratio'] = F0_F2_ratios

        
    features_data['Call Number'] = call_numbers
    features_data['F0 Mean'] = F0_means
    features_data['F0 Std'] = F0_stds
    features_data['F0 Skewness'] = F0_skewnesses
    features_data['F0 Kurtosis'] = F0_kurtosises

    features_data['F0 1st Order Diff'] = F0_fst_order_diffs
    features_data['F1 Mean'] = F1_means
    features_data['F2 Mean'] = F2_means
    features_data['F0-F1 Ratio'] = F0_F1_ratios
    features_data['F0-F2 Ratio'] = F0_F2_ratios


    return f0_features_calls, f0_pyin_lb, features_data



if __name__ == '__main__':

    file = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'
    onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))

    save_features_file = 'chickname_features.csv'
    features_data = pd.DataFrame()
    features_data['onsets_sec']= onsets
    features_data['offsets_sec']=offsets


    ##### 1- Load audio file
    audio_y, sr = lb.load(file, sr=44100)
    audio_fy = ut.bp_filter(audio_y, sr, lowcut=1600, highcut=12700)


    audio = audio_fy
    onsets_sec = onsets
    offsets_sec = offsets
    sr = 44100
    pyin_fmin_hz = 1800
    pyin_fmax_hz = 12500
    pyin_beta = (0.10, 0.10)
    pyin_ths = 100
    pyin_resolution = 1


    f0_features_calls, F0_wholesignal, features_data = compute_f0_features(audio, features_data,sr, hop_length, frame_length, win_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution)
    
    
    F1_Hz = F0_wholesignal*2
    F2_Hz = F0_wholesignal*3
    S = np.abs(lb.stft(y=audio, n_fft=frame_length, hop_length=hop_length))
    visualise_spectrogram_and_harmonics(S, F0_wholesignal, F1_Hz, F2_Hz, sr, hop_length)

    
    


