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


list_files = glob.glob(f'{audio_folder}/**/*.wav', recursive=True)

output_directory="/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Chicks_Onset_Detection_project/grid_search_parameters/results_grid_searches"
# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


which_search = 'evaluation' #' #'input_features'#, 'peak_picking', 'evaluation'

if which_search == 'input_features':
    output_file = os.path.join(output_directory, "superflux_search_input_features.json")
elif which_search == 'peak_picking':
    output_file = os.path.join(output_directory, "superflux_search_peak_picking.json")
elif which_search == 'evaluation':
    output_file = os.path.join(output_directory, "superflux_search_evaluation.json")



#non-changing parameters
frame_window= 0.12
n_fft=2048 * 2
hop_length=1024 // 2



# input feature parameters 
num_bands_range = [15, 24, 32, 60]
num_bands = 15
fmin_range=  [1800, 2000, 2050]
fmin = 2050
fmax_range = [5000, 6000, 8000]
fmax = 8000
lag_range= [1,3,5]
lag = 3
max_size_range = [40, 50, 60]
max_size = 60 # These values have already been selected based on input features grid search on the validaition set!!!!!!

# peak picking parameters
pre_avg_range =[1, 10, 20, 25 ]
pre_avg = 10
post_avg_range =[1, 10, 20, 25]
post_avg =10
pre_max_range = [1, 10, 20, 25]
pre_max = 1
post_max_range = [1, 10, 3]
post_max = 1
delta_range = [0, 0.1, 2, 5]
delta = 0.1
wait_range = [0, 1, 10]
wait = 10 # These values have already been selected based on input features grid search on the validaition set!!!!!!

# evaluation parameters
global_shift_range = [-0.1, -0.05, -0.02,-0.01, 0, 0.01, 0.02, 0.05, 0.1] 
correction_shift = 0
window_range = [0.05, 0.1, 0.5]
eval_window = 0.1




if which_search == 'input_features':
    parameter_combinations = list(product(num_bands_range, fmin_range, fmax_range, lag_range, max_size_range))
elif which_search == 'peak_picking':
    parameter_combinations = list(product(pre_avg_range, post_avg_range, pre_max_range, post_max_range, delta_range, wait_range))
elif which_search == 'evaluation':
    parameter_combinations = list(product(window_range, global_shift_range))


overall_fmeasure_and_parameters = {}

for i in tqdm(range(len(parameter_combinations))):
    if which_search == 'input_features':
        num_bands, fmin, fmax, lag, max_size = parameter_combinations[i]
    elif which_search == 'peak_picking':
        pre_avg, post_avg, pre_max, post_max, delta, wait = parameter_combinations[i]
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

        
        
        
        SPF_predictions_in_seconds, SPF_activation_frames, spec_hop_length, spf_sr = onset_detectors.superflux(filepath, spec_hop_length= hop_length, spec_n_fft = n_fft, spec_window= frame_window,
                                                                                    spec_n_mels= num_bands, spec_fmin= fmin, spec_fmax= fmax, spec_lag= lag , spec_max_size=max_size, 
                                                                                    pp_pre_avg=pre_avg, pp_post_avg=post_avg, pp_pre_max=pre_max, pp_post_max=post_max, pp_threshold=delta, pp_wait=wait, 
                                                                                    visualise_activation=True,)
        
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
    overall_fmeasure_and_parameters[i] = {'f_measure' :overall_fmeasure_in_set,'n_mels': num_bands, 'fmin': fmin, 'fmax': fmax, 'lag': lag, 'max_size': max_size,
                                                                'pre_avg': pre_avg, 'post_avg': post_avg, 'pre_max': pre_max, 'post_max': post_max, 
                                                                'delta': delta, 'wait': wait, 'window': eval_window, 'correction_shift': correction_shift}
                                                   

# Save the results in a JSON file
with open(output_file, "w") as file:
    json.dump(overall_fmeasure_and_parameters, file)



