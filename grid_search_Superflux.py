import glob
from itertools import product
import os
import madmom
import numpy as np
import matplotlib.pyplot as plt
#import mir_eval_modified as mir_eval_new
import pandas as pd
from mir_eval_modified.onset import f_measure
from tqdm import tqdm
import evaluation as eval
import onset_detection_algorithms as onset_detectors
import json




audio_folder = "C:\\Users\\anton\\High_quality_dataset"

output_directory=r"./Grid_search_high_quality_dataset/"
# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

metadata = pd.read_csv("C:\\Users\\anton\\High_quality_dataset\\high_quality_dataset_metadata.csv")





print('**************** Starting grid search on Superflux')
#Superflux grid search:
#non-changing parameters
frame_window= 0.12
n_fft=2048 * 2
hop_length=1024 // 2


output_file = os.path.join(output_directory, "Superflux_eval_onset_correction_th.json")

# # Ranges for Grid search parameters:
n_mels_range = [15, 24, 32, 60]
fmin_range=  [1800, 2000, 2050]
fmax_range = [5000, 6000, 8000]

lag_range= [1,3,5]
max_size_range = [40, 50, 60]
global_shift_range = [-0.1, -0.05, -0.02,-0.01, 0, 0.01, 0.02, 0.05, 0.1]

window_range = [0.1]


parameter_combinations = list(product(n_mels_range, fmin_range, fmax_range, lag_range, max_size_range, window_range, global_shift_range)) 

overall_fmeasure_and_parameters = {}
for i in tqdm(range(len(parameter_combinations))):
    n_mels, fmin, fmax, lag, max_size, eval_window, correction_shift = parameter_combinations[i]

    list_fscores_in_set = []
    list_n_events_in_set = []
    
    
    for filepath in glob.glob(f'{audio_folder}/**/*.wav', recursive=True):


        # Load ground truth
        fname = os.path.basename(filepath)
        fpath_sans_ext = filepath.split(".")[0]
        file_txt = os.path.join(audio_folder, f'{fpath_sans_ext}.txt')
        # print(f"file_txt saved in the following directory: {file_txt}")
        filename = os.path.basename(filepath)
        # print(f'Processing {filename}...')

        # Get ground truth onsets from txt file to compare with algorithm results
        gt_onsets = eval.get_reference_onsets(file_txt)
        # print(f'Ground truth onsets: {len(gt_onsets)}')

        
        
        
        SPF_predictions_in_seconds, SPF_activation_frames, spec_hop_length, spf_sr = onset_detectors.superflux(filepath, spec_hop_length= hop_length, spec_n_fft = n_fft, spec_window= frame_window,
                                                                                    spec_n_mels= n_mels, spec_fmin= fmin, spec_fmax= fmax, spec_lag= lag , spec_max_size= max_size, visualise_activation=True)
        
        exp_start = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['Start_experiment_sec'].values[0]   
        exp_end = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['End_experiment_sec'].values[0]
        
        gt_onsets, SPF_predictions_in_seconds, SPF_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                        gt_onsets, SPF_predictions_in_seconds, SPF_activation_frames, hop_length=spec_hop_length, sr= spf_sr )

        list_n_events_in_set.append(len(gt_onsets))
        
        shifted_predictions = eval.global_shift_correction(SPF_predictions_in_seconds, correction_shift )
        
        fmeasure, precision, recall, _, _, _  = f_measure(gt_onsets, shifted_predictions, window=eval_window)

        file_scores = [fmeasure, precision, recall]
        list_fscores_in_set.append(file_scores[0])

    
    # Compute the average F-measure for the set of files
    overall_fmeasure_in_set = eval.compute_weighted_average(list_fscores_in_set, list_n_events_in_set)

    # Save the overall F-measure and the corresponding parameters
    overall_fmeasure_and_parameters[overall_fmeasure_in_set] = {'n_mels': n_mels, 'fmin': fmin, 'fmax': fmax, 'lag': lag, 'max_size': max_size, 'window': eval_window, 'correction_shift': correction_shift}

    

# Save the results in a JSON file
with open(os.path.join(output_directory, 'Superflux_gridsearch_results.json'), "w") as file:
    json.dump(overall_fmeasure_and_parameters, file)
