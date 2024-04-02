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

# create dictionary with the duration of the calls for each chick

# Path to the folder containing the txt files to be evaluated
audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\Data\\normalised_data_only_inside_exp_window\\Testing_set'

#metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
# save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segmentation_task\\testing_calls'
# if not os.path.exists(save_results_folder):
#     os.makedirs(save_results_folder)

# list_files = glob.glob(os.path.join(audio_folder, "*.wav"))
file= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set\\chick367_d1.wav'

# for file in tqdm(list_files):
   
chick = os.path.basename(file)[:-4]

# Load audio file
#file = lb.ex("humpback")
audio_y, sr = lb.load(file, sr=22050,duration=20.0)
# duration = lb.get_duration(y=audio_y, sr=sr)
duration = 20.0


# Parameters for spectrogram and pitch estimation
frame_length = 2048
hop_length = 512
n_fft = frame_length * 2
win_length = frame_length // 2
# in sonic visualiser audio frame per block is 2048, window lenght is 2

st = time.time()
f0_pyin_lb, voiced_flag, voiced_probs = lb.pyin(audio_y, sr=sr, frame_length=frame_length, hop_length=hop_length, fmin=1000, fmax=10000)
et= time.time()
print('it takes ' + str(et-st)+ ' seconds to run pyin for a file of duration ' + str(duration) + ' seconds')
f0_in_times= lb.times_like(f0_pyin_lb, sr=sr, hop_length=hop_length)
# save f0_pyin_lb to csv
f0_pyin_lb_df = pd.DataFrame(f0_pyin_lb)
# replace nan in dataframe with 0
f0_pyin_lb_df = f0_pyin_lb_df.fillna(0)

f0_pyin_lb_df.to_csv(chick + '_f0_pyin_lb.csv', index=False)
# plot spectrogram with f0_in_times
np.save( chick + '_f0_pyin_lb.npy', f0_pyin_lb)
# print('f0_pyin_lb, voiced_
print('f0_pyin_lb', f0_pyin_lb)

# Plotting
slice_st_sec =20
slice_st_frame = int(slice_st_sec*sr/hop_length)

slice_et_sec = 30
slice_et_frame = int(slice_et_sec*sr/hop_length)



# Compute melspectrogram (makes it easier to see pitch output)
S = lb.feature.melspectrogram(y=audio_y, sr=sr, win_length=win_length, hop_length=hop_length, n_mels=128)
S_db = lb.amplitude_to_db(S, ref=np.max)


f0_slice = f0_pyin_lb[slice_st_frame:slice_et_frame + 1]
f0_slice_times = f0_in_times[slice_st_frame:slice_et_frame + 1]

plt.figure(figsize=(10, 6))
# Plot pitch
plt.plot(f0_in_times, f0_pyin_lb, label='Pitch (Hz)', color='red')
# Plot Mel spectrogram
lb.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', alpha=0.75)
plt.colorbar(format='%+2.0f dB')
plt.title('Pitch and Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()