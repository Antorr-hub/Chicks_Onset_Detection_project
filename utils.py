import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import soundfile as sf

#This  function was built to calculate the duration of the calls from the ground truth onsets and offsets
def calculate_durations(events):   
    durations = []
    for event in events:
        duration = event[1] - event[0]
        durations.append(duration)
    return durations







def save_spec_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        plt.figure(figsize=(6, 4))
        plt.imshow(segment, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()




# This function was built to segment the calls using onsets and offsets and saving the spectrogram segments

def get_calls_spec(filename, onsets, offsets, save_folder):
    # Load the audio file
    y, sr = lb.load(filename, sr=44100)

    # Initialize lists to store spectrogram slices and their corresponding filenames
    calls = []
    call_filenames = []

    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Convert time (in seconds) to sample indices
        onset_frame = lb.time_to_samples(onset, sr=44100)
        offset_frame = lb.time_to_samples(offset, sr=44100)

        # Extract the spectrogram slice from onset to offset
        call_audio = y[onset_frame: offset_frame]
        
        # Compute the linear spectrogram of the call
        D = np.abs(lb.stft(call_audio, n_fft=2048, hop_length=512))

        # Apply logarithmic transformation to the spectrogram
        log_D = lb.amplitude_to_db(D, ref=np.max)

        # Optionally rescale the log-spectrogram for visualization
        min_db = np.min(log_D)
        max_db = np.max(log_D)
        scaled_log_D = (log_D - min_db) / (max_db - min_db)

        # Append the scaled log-spectrogram slice to the calls list
        calls.append(scaled_log_D)

        # Define filename for saving the spectrogram segment
        call_filename = os.path.join(save_folder, f"segment_{i}.png")
        call_filenames.append(call_filename)

        # Save the spectrogram segment
        plt.figure(figsize=(6, 4))
        plt.imshow(scaled_log_D, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(call_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    return calls, call_filenames



def save_waveform_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save the waveform segment to a .wav file
        sf.write(filename, segment, 44100)

def get_calls_waveform(filename, onsets, offsets, save_results_folder):
    # Load the audio file
    y, sr = lb.load(filename, sr=44100)

    # Initialize an empty list to store waveform segments and segment filenames
    calls = []
    segment_filenames = []

    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Convert time (in seconds) to sample indices
        onset_frame = lb.time_to_samples(onset, sr=sr)
        offset_frame = lb.time_to_samples(offset, sr=sr)

        # Extract the waveform segment from onset to offset
        call_waveform = y[onset_frame:offset_frame]

        # Append the waveform segment to the calls list
        calls.append(call_waveform)

        # Define filename for saving the waveform segment
        segment_filename = os.path.join(save_results_folder, f"{os.path.basename(filename)}_segment_{i}.wav")
        segment_filenames.append(segment_filename)

    # Save waveform segments
    save_waveform_segments(calls, segment_filenames)

    return calls

