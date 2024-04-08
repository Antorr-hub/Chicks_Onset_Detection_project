import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal



#This  function was built to calculate the duration of the calls from the ground truth onsets and offsets
def calculate_durations(events):   
    durations = []
    for event in events:
        duration = event[1] - event[0]
        durations.append(duration)
    return durations




def bp_filter(audio_y, sr=44100, lowcut=2050, highcut=8000):
    # Apply the bandpass filter
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
def get_calls_spec( audio_y , onsets, offsets, save_folder, sr=44100, hop_length=512, n_fft=2048*2):

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





def save_F0_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save a csv file with the F0 values
        segment.to_csv(filename, index=False)


  


# def get_calls_F0(F0_in_times, F0_in_frames, onsets, offsets):
#     # Load the audio file

#     sr= 22050
#     # Initialize an empty list to store waveform segments and segment filenames
#     calls = []
#     segment_filenames = []

#     # Loop through each onset and offset pair
#     for i, (onset, offset) in enumerate(zip(onsets, offsets)):
#         # Convert time (in seconds) to sample indices
#         # onset_frame = lb.time_to_frames(onset, sr=sr)
#         # offset_frame = lb.time_to_frames(offset, sr=sr)

#         # Extract the waveform segment from onset to offset
#         call_F0 = F0_in_times[onset:offset]

#         # Append the waveform segment to the calls list
#         calls.append(call_F0)

#         # # Define filename for saving the waveform segment
#         # segment_filename = os.path.join(save_results_folder, f"{os.path.basename(filename)}_segment_{i}.wav")
#         # segment_filenames.append(segment_filename)

#     # Save segments
#     #save_F0_segments(calls, segment_filenames)

#     return calls
        

def get_calls_F0(F0_in_times, F0_in_frames, onsets, offsets):
    # Initialize an empty list to store F0 slices
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Find the indices corresponding to the onset and offset times
        onset_index = lb.time_to_frames(onset, sr=22050)
        offset_index = lb.time_to_frames(offset, sr=22050)

        # Extract the F0 values within the specified time window
        call_F0 = F0_in_frames[(F0_in_times >= onset) & (F0_in_times <= offset)]

        # Append the F0 slice to the calls list
        calls.append(call_F0)

    return calls




