import glob
from itertools import product
import os
import pandas as pd
import madmom
import numpy as np
import matplotlib.pyplot as plt
#import mir_eval_modified as mir_eval_new

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


# Normalised weighted phase deviation grid search:
print('********************* Starting grid search on Normalised weighted phase deviation')

output_file = os.path.join(output_directory, "NWPD.json")
#non-changing parameters
hop_length= 441
sr= 44100
# Ranges for Grid search parameters for peak picking with NWPD  
threshold_range = [0.8, 0.92]#[0.8, 0.92, 0.95]   #threshold=0.92, smooth=None, pre_avg=0, post_avg=0, pre_max=30, post_max=30)
pre_avg_range =[0]#[0, 1, 10]
post_avg_range =[0]#[0, 1, 10]
pre_max_range = [20]#[20, 30, 50]
post_max_range = [20]#[20, 30, 50]
global_shift_range = [-0.1]# [-0.1, -0.05, -0.02,-0.01, 0, 0.01, 0.02, 0.05, 0.1]

window_range = [0.1]


parameter_combinations = list(product(threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range, window_range, global_shift_range))


overall_fmeasure_and_parameters = {}
for i in tqdm(range(len(parameter_combinations))):
    threshold, pre_avg, post_avg, pre_max, post_max, eval_window, correction_shift = parameter_combinations[i]
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
        
        
        NWPD_predictions_in_seconds, NWPD_activation_frames = onset_detectors.normalized_weighted_phase_deviation(filepath, hop_length=441, sr=44100, pp_threshold= threshold, pp_pre_avg=pre_avg, pp_post_avg=post_avg, 
                                    pp_pre_max=pre_max , pp_post_max=post_max, visualise_activation=True)

        
        exp_start = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['Start_experiment_sec'].values[0]   
        exp_end = metadata[metadata['Filename'] == os.path.basename(filepath)[:-4]]['End_experiment_sec'].values[0]
        
        gt_onsets, NWPDpredictions_in_seconds, NWPD_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                        gt_onsets, NWPD_predictions_in_seconds, NWPD_activation_frames, hop_length=441, sr=44100 )
        list_n_events_in_set.append(len(gt_onsets))
        
        shifted_predictions = eval.global_shift_correction(NWPDpredictions_in_seconds, correction_shift )
        
        fmeasure, precision, recall, _, _, _  = f_measure(gt_onsets, shifted_predictions, window=eval_window)

        file_scores = [fmeasure, precision, recall]
        list_fscores_in_set.append(file_scores[0])

    
    # Compute the average F-measure for the set of files
    overall_fmeasure_in_set = eval.compute_weighted_average(list_fscores_in_set, list_n_events_in_set)

    # Save the overall F-measure and the corresponding parameters
    overall_fmeasure_and_parameters[overall_fmeasure_in_set] = {'threshold': threshold, 'pre_avg': pre_avg, 'post_avg': post_avg, 'pre_max': pre_max, 'post_max': post_max, 'window': eval_window, 'correction_shift': correction_shift}

    

# Save the results in a JSON file
with open(os.path.join(output_directory, 'NWPD_gridsearch_results.json'), "w") as file:
    json.dump(overall_fmeasure_and_parameters, file)




