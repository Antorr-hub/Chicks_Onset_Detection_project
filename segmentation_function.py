import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

def segmentation_calls_spec(filename, onsets, offsets):
    # Load the audio file
    y, sr = lb.load(filename, sr=44100)

    # Initialize an empty list to store spectrogram slices
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frame = lb.time_to_frames(onset, sr=44100, hop_length=512)
        offset_frame = lb.time_to_frames(offset, sr=44100, hop_length=512)

        # Extract the spectrogram slice from onset to offset
        call_spectrogram = lb.feature.melspectrogram(y=y[onset_frame: offset_frame], sr=sr, hop_length=512, n_fft=2048 * 2, window= 0.15, fmin=2050, fmax=8000, n_mels=15)
        
        # Append the spectrogram slice to the calls list
        calls.append(call_spectrogram)

    return calls







def segmentation_calls_waveform(filename, onsets, offsets):
    # Load the audio file
    y, sr = lb.load(filename, sr=44100)

    # Initialize an empty list to store waveform segments
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        # Convert time (in seconds) to sample indices
        onset_frame = lb.time_to_frames(onset, sr=44100, hop_length=512)
        offset_frame = lb.time_to_frames(offset, sr=44100, hop_length=512)

        # Extract the waveform segment from onset to offset
        call_waveform = y[onset_frame :offset_frame]
        
        # Append the waveform segment to the calls list
        calls.append(call_waveform)

    return calls




def save_waveform_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        lb.output.write_wav(filename, segment, sr=44100)

# # Usage
# segments = [...]  # List of waveform segments
# filenames = [...]  # List of filenames to save the segments
# save_waveform_segments(segments, filenames)




def save_spectrogram_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        plt.figure(figsize=(6, 4))
        plt.imshow(segment, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

# Usage
segments = [...]  # List of spectrogram segments
filenames = [...]  # List of filenames to save the segments
save_spectrogram_segments(segments, filenames)