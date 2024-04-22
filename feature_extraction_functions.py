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
    plt.figure(figsize=(13, 8))
    lb.display.specshow(lb.amplitude_to_db(spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
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
    plt.show()
    
    return





def visualise_spectrogram_and_spectral_centroid(lin_spec, spectral_centroid_feature_calls, sr, hop_length):
    # Plot the linear spectrogram
    plt.figure(figsize=(13, 8))
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    times = lb.times_like(spectral_centroid_feature_calls[0], sr=sr, hop_length=hop_length)
    plt.plot(times, spectral_centroid_feature_calls[0], label='Spectral Centroid', color='green')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Linear Spectrogram and Spectral Centroid')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return  


def visualise_spectrogram_and_RMS(lin_spec, rms_features_calls, sr, hop_length):
    # Plot the linear spectrogram
    fig, ax = plt.subplots(nrows=2, sharex=True)
    rms_y_times = lb.times_like(rms_features_calls, sr=sr, hop_length=hop_length)
    ax[0].semilogy(rms_y_times, rms_features_calls, label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    ax[1].set(xlabel='Time (s)', ylabel='Frequency')
    plt.show()

    return


def spectral_centroid(audio_fy, features_data, sr, frame_length, hop_length):

    ''' Compute the mean of the spectral centroid for each call in the audio file
    returns: return the mean of the spectral centroid for each call in the audio file
    '''
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec

   
    Spectral_Centroid_mean = [] 
    call_numbers = []
    # Compute the spectrogram of the entire audio file
    lin_spec= np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))
        
    # Extract the spectrogram calls from the audio file
    calls_s_files = ut.segment_spectrogram(spectrogram= lin_spec, onsets=onsets_sec, offsets=offsets_sec, sr=sr)
      

    for i, call_s in enumerate(calls_s_files):
        call_numbers.append(i)


        # Compute the spectral centroid and then extract the mean
        spectral_centroid_call = lb.feature.spectral_centroid(S=call_s, sr=sr, n_fft=frame_length, hop_length=hop_length)

        # deal with nans
        spectral_centroid_without_nans = spectral_centroid_call[~np.isnan(spectral_centroid_call)]
        # replace nans with zeros
        if spectral_centroid_without_nans.size == 0:
            Spectral_Centroid_mean.append(np.nan)
            continue 
        else:     
        # compute the mean of the spectral centroid
            mean_spectral_centroid = np.mean(spectral_centroid_without_nans)
            Spectral_Centroid_mean.append(mean_spectral_centroid)
    
    spectral_centroid_feature_calls = pd.DataFrame()
    spectral_centroid_feature_calls['Spectral Centroid Mean'] = Spectral_Centroid_mean

    features_data['Spectral Centroid Mean'] = Spectral_Centroid_mean

    return lin_spec, spectral_centroid_feature_calls, features_data







def rms_features(audio_fy, features_data, sr, frame_length, hop_length):
    ''' Compute the mean and the standard deviation of the RMS of each call in the audio file'''

    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec


    call_numbers = []
    RMS_mean = []
    RMS_std = []

    # Extract the waveform segments from the audio file
    calls_wave_file = ut.get_calls_waveform(audio_fy, onsets_sec, offsets_sec, sr)

    for i, call in enumerate(calls_wave_file):
        call_numbers.append(i)        
        # Compute the RMS of each call
        rms_call = lb.feature.rms(y=call, frame_length=frame_length, hop_length=hop_length)
        rms_call_without_nans = rms_call[~np.isnan(rms_call)]
        if rms_call_without_nans.size == 0:
            RMS_mean.append(np.nan)
            RMS_std.append(np.nan)
            continue
        else:
            mean_rms = np.mean(rms_call_without_nans)
            st_dev_rms = np.std(rms_call_without_nans)
            RMS_mean.append(mean_rms)
            RMS_std.append(st_dev_rms)

    rms_features_calls = pd.DataFrame()
    rms_features_calls['RMS Mean'] = RMS_mean
    rms_features_calls['RMS Std'] = RMS_std

    features_data['RMS Mean'] = RMS_mean
    features_data['RMS Std'] = RMS_std

    return rms_call, rms_features_calls, features_data



def compute_envelope(audio_fy, features_data, sr):

    envelope_mean = []
    envelope_std = []
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec

    calls_wave_file = ut.get_calls_waveform(audio_fy, onsets_sec, offsets_sec, sr)

    for i, call in enumerate(calls_wave_file):
        envelope = np.abs(hilbert(call))
        envelope_mean.append(np.mean(envelope))
        envelope_std.append(np.std(envelope))

    envelope_features_calls = pd.DataFrame()
    envelope_features_calls['Envelope Mean'] = envelope_mean
    envelope_features_calls['Envelope Std'] = envelope_std

    features_data['Envelope Mean'] = envelope_mean
    features_data['Envelope Std'] = envelope_std

    return envelope_features_calls, features_data








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
        call_numbers.append(i)
        f0_call_without_nans = f0_call[~np.isnan(f0_call)]
        if f0_call_without_nans.size == 0:
            F0_means.append(np.nan)
            F0_stds.append(np.nan)
            F0_skewnesses.append(np.nan)
            F0_kurtosises.append(np.nan)
            F0_fst_order_diffs.append(np.nan)
            F1_means.append(np.nan)
            F2_means.append(np.nan)
            F0_F1_ratios.append(np.nan)
            F0_F2_ratios.append(np.nan)
            continue
        else:
            # compute the statistics
            f0_call_mean = f0_call_without_nans.mean()
            f0_call_std = f0_call_without_nans.std()
            f0_call_skewness = stats.skew(f0_call_without_nans)
            f0_call_kurtosis = stats.kurtosis(f0_call)
            
            
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

    filename = os.path.basename(file)[:-4]

    save_features_file = 'features_data_' + filename + '.csv'   

    save_folder= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\features'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_features_file = os.path.join(save_folder, save_features_file)


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
    pyin_resolution = 0.02


    #f0_features_calls, F0_wholesignal, features_data = compute_f0_features(audio, features_data,sr, hop_length, frame_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution)
    
    # # save locally features_data to a csv file
    # features_data.to_csv(save_features_file, index=False)
    
    # F1_Hz = F0_wholesignal*2
    # F2_Hz = F0_wholesignal*3


    # lin_spec, spectral_centroid_feature_calls, features_data = spectral_centroid(audio_fy, features_data, sr, frame_length, hop_length)
    # # save locally features_data to a csv file
    # features_data.to_csv(save_features_file, index=False)


    rms_call, rms_features_calls, features_data = rms_features(audio_fy, features_data, sr, frame_length, hop_length)
    
    # save locally features_data to a csv file

    features_data.to_csv(save_features_file, index=False)


    # envelope_features_calls, features_data = compute_envelope(audio_fy, features_data, sr, frame_length, hop_length)

    # features_data.to_csv(save_features_file, index=False)


    # Visualise the spectrogram and the features computed (F0 stats, F0/F1 ratio, F0/F2 ratio, spectral centroid stats, RMS stats, envelope stats)

    S = np.abs(lb.stft(y=audio, n_fft=frame_length, hop_length=hop_length))

    #visualise_spectrogram_and_harmonics(S, F0_wholesignal, F1_Hz, F2_Hz, sr, hop_length)
    
    # visualise_spectrogram_and_spectral_centroid(S, spectral_centroid_feature_calls=spectral_centroid_feature_calls, sr=sr, hop_length=hop_length)
    
    #visualise_spectrogram_and_RMS(S, rms_features_calls=rms_features_calls, sr=sr, hop_length=hop_length)