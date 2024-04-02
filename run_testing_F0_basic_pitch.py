from basic_pitch.inference import predict_and_save
import tensorflow as tf
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import matplotlib.pyplot as plt
import os
import pretty_midi
import numpy as np
import librosa.display as display
import librosa.display as lb
# List of input audio paths

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
# Example librosa file
#fpath = "C:/Users/anton/Chicks_Onset_Detection_project/Data/normalised_data_only_inside_exp_window/Testing_set/chick367_d1.wav"
fpath = lb.ex("humpback")
model_output, midi_data, note_events = predict(fpath, basic_pitch_model)

# Output directory
output_directory = "basic-pitch C:/Users/anton/Chicks_Onset_Detection_project/result_basic_pitch"
chick = os.path.basename(fpath)[:-4]


# Extract note information
notes = []
for instrument in midi_data.instruments:
    for note in instrument.notes:
        notes.append((note.start, note.pitch))

# Sort notes by start time
notes.sort(key=lambda x: x[0])

# Convert MIDI notes to frequencies in Hz
hz_notes = [(2 ** ((note[1] - 69) / 12)) * 440 for note in notes]

y, sr = librosa.load(fpath)

# Compute spectrogram
S = librosa.stft(y=y, sr=sr, win_length=2048, hop_length=512)
spec = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# Get time points for plotting
times = librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram with MIDI Notes')

# Superimpose MIDI note events
# Basic_pitch gives polyphonic pitch output. So its easier to display the note event
# as scatter plots.
for i, note_time in enumerate(notes):
    plt.scatter(note_time[0], hz_notes[i], color='red', marker='o')

plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()