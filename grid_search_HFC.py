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








# audio_folder = "C:\\Users\\anton\\High_quality_dataset"
audio_folder = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Validation_set"
metadata = pd.read_csv(os.path.join(audio_folder, "chicks_validation_metadata.csv"))

output_directory="/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Chicks_Onset_Detection_project/grid_search_parameters/results_grid_searches"
# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


which_search = 'peak_picking' #' #'input_features'#, 'peak_picking', 'evaluation'

if which_search == 'input_features':
    output_file = os.path.join(output_directory, "HFC_search_input_features.json")
elif which_search == 'peak_picking':
    output_file = os.path.join(output_directory, "HFC_search_peak_picking.json")
elif which_search == 'evaluation':
    output_file = os.path.join(output_directory, "HFC_search_evaluation.json")


#non-changing parameters
hop_length= 441
sr= 44100

# Ranges for Grid search paramseters for hfc algorithm
num_bands_range =[15, 32, 60]
num_bands = 15
fmin_range=  [ 1800, 2000, 2500]
fmin = 2500
fmax_range = [5000, 6000, 8000]
fmax = 5000
fref_range = [2000, 2500, 2800]
fref = 2800        # These values have already been selected based on input features grid search on the validaition set

    
# peak picking parameters
threshold_range = [1.8, 2.5, 3, 3.5]  
threshold = 1.8
pre_avg_range =[20, 25, 30, 40]
pre_avg = 25
post_avg_range =[20, 25, 30, 40]
post_avg = 25
pre_max_range = [1, 2, 3]
pre_max = 1
post_max_range = [1, 2, 3]
post_max = 1

# evaluation parameters
global_shift_range = [-0.1, -0.05, -0.02,-0.01, 0, 0.01, 0.02, 0.05, 0.1] 
correction_shift = 0
window_range = [0.05, 0.1, 0.2, 0.5, 2]
eval_window = 0.1


list_files = glob.glob(f'{audio_folder}/**/*.wav', recursive=True)

if which_search == 'input_features':
    parameter_combinations = list(product(num_bands_range, fmin_range, fmax_range, fref_range))
elif which_search == 'peak_picking':
    parameter_combinations = list(product(threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range))
elif which_search == 'evaluation':
    parameter_combinations = list(product(window_range, global_shift_range))


overall_fmeasure_and_parameters = {}

for i in tqdm(range(len(parameter_combinations))):
    if which_search == 'input_features':
        num_bands, fmin, fmax, fref = parameter_combinations[i]
    elif which_search == 'peak_picking':
        threshold, pre_avg, post_avg, pre_max, post_max = parameter_combinations[i]
    elif which_search == 'evaluation':
        eval_window, correction_shift = parameter_combinations[i]

    list_fscores_in_set = []
    list_n_events_in_set = []
    
    
    for filepath in list_files:


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

        HFC_predictions_in_seconds, HFC_activation_frames = onset_detectors.high_frequency_content(filepath, hop_length=441, sr=44100, spec_num_bands=num_bands, spec_fmin=fmin, spec_fmax=fmax, spec_fref=fref,
                                    pp_threshold=threshold, pp_pre_avg=pre_avg, pp_post_avg=post_avg,pp_pre_max=pre_max , pp_post_max=post_max, visualise_activation=True)
                                                                                                   

        exp_start = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['Start_experiment_sec'].values[0]   
        exp_end = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['End_experiment_sec'].values[0]
        
        gt_onsets, HFC_predictions_in_seconds, HFC_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                        gt_onsets, HFC_predictions_in_seconds, HFC_activation_frames, hop_length=441, sr=44100 )
        list_n_events_in_set.append(len(gt_onsets))
        
        shifted_predictions = eval.global_shift_correction(HFC_predictions_in_seconds, correction_shift )
        
        fmeasure, precision, recall, _, _, _  = f_measure(gt_onsets, shifted_predictions, window=eval_window)

        file_scores = [fmeasure, precision, recall]
        list_fscores_in_set.append(file_scores[0])

    
    # Compute the average F-measure for the set of files
    overall_fmeasure_in_set = eval.compute_weighted_average(list_fscores_in_set, list_n_events_in_set)

        # Save the overall F-measure and the corresponding parameters
    overall_fmeasure_and_parameters[i] = {'f_measure' :overall_fmeasure_in_set, 'num_bands': num_bands, 'fmin': fmin, 'fmax': fmax, 'fref': fref,
                                           'threshold': threshold, 'pre_avg': pre_avg, 'post_avg': post_avg, 'pre_max': pre_max, 'post_max': post_max, 
                                           'window': eval_window, 'correction_shift': correction_shift}

# Save the results in a JSON file
with open(output_file, "w") as file:
    json.dump(overall_fmeasure_and_parameters, file)

print('stop')



