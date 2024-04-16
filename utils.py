import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from librosa.core import fft_frequencies
from librosa.util.exceptions import ParameterError



#This  function was built to calculate the duration of the calls from the ground truth onsets and offsets
def calculate_durations(events): 
    '''This function calculates the duration of each event in a list of events.
    '''  
    durations = []
    for event in events:
        duration = event[1] - event[0]
        durations.append(duration)
    return durations




def bp_filter(audio_y, sr=44100, lowcut=2050, highcut=8000):
    # Apply the bandpass filter
    '''Here, the Nyquist frequency is used to normalise the filter cut-off frequencies
      with respect to the sampling rate of the audio signal. This is important because 
      digital filters operate at normalised frequencies, so it is necessary to express 
      the filter's cut-off frequencies in relation to the Nyquist frequency to ensure 
      that the filter works correctly with the sampled audio signal.
      '''
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio_y = signal.filtfilt(b, a, audio_y)

    return audio_y





def save_spec_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        plt.figure(figsize=(4, 2))
        plt.imshow(segment, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()




# This function was built to segment the calls using onsets and offsets and saving the spectrogram segments
def get_calls_spec(audio_y , onsets, offsets, save_folder, sr=44100, hop_length=512, n_fft=2048*2):
    '''Extracts spectrogram segments from an audio signal based on onset and offset times.'''
    # compute the duration of the audio file
    duration= lb.get_duration(audio_y, sr=sr)

    # Initialize lists to store spectrogram slices and their corresponding filenames
    calls_S = []
    call_filenames = []

    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Convert time (in seconds) to sample indices
        onset_sample = lb.time_to_samples(onset, sr=sr)
        offset_sample = lb.time_to_samples(offset, sr=sr)

        #Extract the spectrogram slice from onset to offset
        epsilon = duration*0.01
        epsilon_samples = lb.time_to_samples(epsilon, sr=44100)

        call_audio = audio_y[onset_sample: offset_sample + epsilon_samples]

        # Compute the linear spectrogram of the call
        D = np.abs(lb.stft(call_audio, n_fft=n_fft, hop_length=hop_length))

        # Apply logarithmic transformation to the spectrogram
        log_D = lb.amplitude_to_db(D, ref=np.max)

        # Optionally rescale the log-spectrogram for visualization
        min_db = np.min(log_D)
        max_db = np.max(log_D)
        scaled_log_D = (log_D - min_db) / (max_db - min_db)

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(scaled_log_D)

        # # Define filename for saving the spectrogram segment
        # call_filename = os.path.join(save_folder, f"segment_{i}.png")
        # call_filenames.append(call_filename)

        # # Save the spectrogram segment
        # plt.figure(figsize=(35, 7))
        # plt.imshow(scaled_log_D, aspect='auto', origin='lower')
        # plt.axis('off')
        # plt.savefig(call_filename, bbox_inches='tight', pad_inches=0)
        # plt.close()
        #save_spec_segments(calls, call_filenames)
    return calls_S



def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):

    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_sample = lb.time_to_frames(onset, sr=sr)
        offset_sample = lb.time_to_frames(offset, sr=sr)

        #Extract the spectrogram slice from onset to offset 
        # REVIEW THIS value of epsilon
        # epsilon = duration*0.0001
        # epsilon_samples = lb.time_to_samples(epsilon, sr=44100)

        call_spec = spectrogram[:, onset_sample: offset_sample]#+ epsilon_samples]

        # save the spectrogram slice
        # # # # Compute Mel spectrogram
        # S = lb.feature.melspectrogram(y=call_spec, sr=sr, hop_length=hop_length, n_mels=128)
        # S_db = lb.amplitude_to_db(S, ref=np.max)
        # # Plot Mel spectrogram
        # plt.figure(figsize=(7, 6))
        # lb.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', alpha=0.75)
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel spectrogram')
        # plt.tight_layout()
        # plt.show()

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
    
    return calls_S
    

    



def spectral_centroid(call_spec, sr=44100, n_fft= 2048*2, hop_length= 512):
    # Compute the spectral centroid as centroid  
    
    # [t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    freqs = lb.fft_frequencies(sr=sr, n_fft=n_fft)
    centroid = np.sum(freqs * call_spec, axis=0) / np.sum(call_spec, axis=0)
    return centroid
    











def save_waveform_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save the waveform segment to a .wav file
        sf.write(filename, segment, 44100)
        

def get_calls_waveform(audio_y, onsets, offsets, sr=44100):
    '''
    Extracts waveform segments from an audio signal 
    based on onset and offset times.
    '''
    calls = []
    segment_filenames = []

    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Convert time (in seconds) to sample indices
        onset_frame = lb.time_to_samples(onset, sr=sr)
        offset_frame = lb.time_to_samples(offset, sr=sr)

        # Extract the waveform segment from onset to offset
        call_waveform = audio_y[onset_frame:offset_frame]

        # Append the waveform segment to the calls list
        calls.append(call_waveform)

    return calls





def save_F0_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save a csv file with the F0 values
        segment.to_csv(filename, index=False)



        

def get_calls_F0(F0_in_frames, onsets, offsets):
    # Initialize an empty list to store F0 slices
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Find the indices corresponding to the onset and offset times
        onset_index = lb.time_to_frames(onset, sr=22050)
        offset_index = lb.time_to_frames(offset, sr=22050)

        # Extract the F0 values within the specified time window
        call_F0 = F0_in_frames[onset_index:offset_index]
        
        # Append the F0 slice to the calls list
        calls.append(call_F0)

    return calls



def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None):
    """Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        Audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        Audio sampling rate of `y`
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) Spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        Hop length for STFT.
    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.

    Returns
    -------
    centroid : np.ndarray [shape=(..., 1, t)]
        Centroid frequencies
    """

    # Check if spectrogram is provided, otherwise compute it
    if S is None:
        S = np.abs(lb.stft(y=y, n_fft=n_fft, hop_length=hop_length))

    if not np.isrealobj(S):
        raise ParameterError("Spectral centroid is only defined with real-valued input")
    elif np.any(S < 0):
        raise ParameterError("Spectral centroid is only defined with non-negative energies")

    # Compute the center frequencies of each bin
    if freq is None:
        freq = lb.fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        # Reshape for broadcasting
        freq = np.expand_dims(freq, axis=-1)

    # Column-normalize S
    denominator = np.maximum(np.sum(S, axis=-2, keepdims=True), np.finfo(float).eps)
    centroid = np.sum(freq * (S / denominator), axis=-2, keepdims=True)

    return centroid
