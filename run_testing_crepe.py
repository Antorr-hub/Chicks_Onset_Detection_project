import essentia.standard as estd
import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa.display as display
imporrt crepe as crepe
 
# Load audio file
fpath = librosa.ex("humpback")

# Loading first 20 seconds for faster debugging
y, sr = librosa.load(fpath)

st= time.time 
crepe = estd.PitchCREPE(batchSize=1, hopSize=200)
f0, confidence = crepe(y)
et= time.time()

print('it takes ' + str(et-st)+ ' seconds to run Basic pitch for a file of duration ' + str(duration) + ' seconds')
# Compute melspectrogram (makes it easier to see pitch output)
S = librosa.feature.melspectrogram(y=y, sr=sr, win_length=2048, hop_length=512, n_mels=128)
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Get time axis for plotting
times = librosa.times_like(f0, sr=22050, hop_length=512)

plt.figure(figsize=(10, 6))

# Plot pitch
plt.plot(times, f0, label='Pitch (Hz)', color='red')

# Plot Mel spectrogram
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', alpha=0.75)
plt.colorbar(format='%+2.0f dB')
plt.title('Pitch and Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.legend()