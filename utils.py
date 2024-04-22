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




def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):

    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        #Extract the spectrogram slice from onset to offset 
        # REVIEW THIS value of epsilon
        # epsilon = duration*0.0001
        # epsilon_samples = lb.time_to_samples(epsilon, sr=44100)

        call_spec = spectrogram[:, onset_frames: offset_frames]#+ epsilon_samples]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
    
    return calls_S
    

    



def spectral_centroid(call_spec, sr=44100, n_fft= 2048*2, hop_length= 512):
    # Compute the spectral centroid as centroid  
    
    # [t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    freqs = lb.fft_frequencies(sr=sr, n_fft=n_fft)
    centroid = np.sum(freqs * call_spec, axis=0) / np.sum(call_spec, axis=0)
    return centroid
    



def get_calls_waveform(audio_y, onsets, offsets, sr=44100):
    '''
    Extracts waveform segments from an audio signal 
    based on onset and offset times.
    '''
    calls = []
    segment_filenames = []

    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Convert time (in seconds) to frame indices
        onset_frame = lb.time_to_frames(onset, sr=sr)
        offset_frame = lb.time_to_frames(offset, sr=sr)

        # Extract the waveform segment from onset to offset
        call_waveform = audio_y[onset_frame:offset_frame]

        # Append the waveform segment to the calls list
        calls.append(call_waveform)

    return calls





def save_F0_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save a csv file with the F0 values
        segment.to_csv(filename, index=False)



        

def get_calls_F0(F0_in_frames, onsets, offsets, sr=44100, hop_length=512, n_fft=2048*2):
    # Initialize an empty list to store F0 slices
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Find the indices corresponding to the onset and offset times
        onset_index = lb.time_to_frames(onset, sr=sr, hop_length=hop_length, n_fft=n_fft)
        offset_index = lb.time_to_frames(offset, sr=sr, hop_length=hop_length, n_fft=n_fft)

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
