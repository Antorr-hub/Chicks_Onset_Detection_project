import os
import glob
from tqdm import tqdm
import numpy as np
import librosa as lb
import pandas as pd
import evaluation as my_eval
from offset_detection import offset_detection_on_spectrograms
from mir_eval_modified.onset import f_measure
import json
import utils as ut


# create dictionary with the duration of the calls for each chick

# Path to the folder containing the txt files to be evaluated
audio_folder = 'C:\\Users\\anton\\Data_normalised\\Testing_set'

metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")



save_results_folder = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Segments_testing_calls'
if not os.path.exists(save_results_folder):
    os.makedirs(save_results_folder)


segments_calls = []
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))


for file in tqdm(list_files):
   
    chick = os.path.basename(file)[:-4]

    # create folder for each chick results
    chick_folder = os.path.join(save_results_folder, chick)
    if not os.path.exists(chick_folder):
            os.mkdir(chick_folder)

    exp_start = metadata[metadata['Filename'] == chick]['Start_experiment_sec'].values[0]   
    exp_end = metadata[metadata['Filename'] == chick]['End_experiment_sec'].values[0]
    # # get ground truth (onsets, offsets)
    gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    gt_offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))

    # discard events outside the experiment window
    new_gt_onsets, new_gt_offsets = my_eval.discard_events_outside_experiment_window_double_threshold(exp_start,exp_end,gt_onsets, gt_offsets)
   
    # get the waveform segments for each call
    chick_calls_wavs= ut.get_calls_waveform(file, new_gt_onsets, new_gt_offsets, save_results_folder)

    #chick_calls_spec = ut.get_calls_spec(chick, new_gt_onsets, new_gt_offsets)

    # Append the waveform segments to the segments_calls list
    segments_calls.append(chick_calls_wavs)
    #chick_calls['calls_spec'] = chick_calls_spec
