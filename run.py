from algorithms import *
#from utils import *

from scipy import signal
import pandas as pd
import mir_eval
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
import os
import json
import glob
import librosa
import numpy as np
import pandas as pd

def get_reference_onsets(file_txt):
    gt_onsets = []
    with open(file_txt, "r",  encoding='latin-1') as file:
            rows = file.readlines()

    for row in rows:
        columns = row.split()

        if columns:
            first_value = float(columns[0])
            gt_onsets.append(first_value)

    print("Ground truth onset:", len(gt_onsets))
   
    if not gt_onsets:
        print("File cannot be read!")
    else:
        print("No onsets saved.")        
    return gt_onsets

def evaluate_algorithm (reference_file, algorithm_results,  window= 0.50):
    # Load ground truth 
    ground_truth_onsets = mir_eval.io.load_events(reference_file)
    # Load algorithm results
    predicted_onsets = mir_eval.io.load_events(algorithm_results)
    # Evaluate and return scores
    scores = mir_eval.onset.evaluate(ground_truth_onsets,  predicted_onsets, window=window)
    #return scores
    return scores


def main():

    #cfg = load_yaml('config.yaml')

    root = os.getcwd()
    data_dir = os.path.join(root,"Data")
    audio_dir = os.path.join(data_dir,"Training_set")
    gt_dir = os.path.join(data_dir,"Training_set")
    result_dir = os.path.join(root,"Results")
    output_gt_dir = os.path.join(root,"Outputs_gt")
    out_dir= os.path.join(root,"Onset_Outputs")
    plot_dir = os.path.join(root,"Onsets_plots")

    # Dictionary of algorithms
    algorithm_dict = { 
        'High_frequency_content' : hfc_ons_detect,
        'Threshold_phase_deviation' : tpd_ons_detect,
        'Norm_weight_phase_deviation' : nwpd_ons_detect,
        'Rectified_complex_domain' : rcd_ons_detect,
        'Superflux' : superflux_ons_detect,
        'Double_threshold_onset' : double_thr_ons_detect
    }

    # Create output directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Create json file to store results


    for fpath in glob.glob(f'{audio_dir}/**/*.wav', recursive=True):
        
        # Load audio file
        # y = get_audio(fpath)
        print(f" The file loaded is: {fpath}")
        # Load ground truth
        fname = os.path.basename(fpath)
        fpath_sans_ext = fpath.split(".")[0]
        print(f"The file that has been processed is: {fpath_sans_ext}")
        file_txt = os.path.join(gt_dir, f'{fpath_sans_ext}.txt')
        print(f"file_txt: {file_txt}")



        # Get ground truth onsets from txt file to compare with algorithm results
        gt_onsets = get_reference_onsets(file_txt)
        print(f'Processing {fpath}...')
        print(f'Ground truth onsets: {len(gt_onsets)}')
        # Load ground truth file for evaluation
        gt = os.path.join(output_gt_dir, f'{fpath_sans_ext}.txt')
    
        json_file = os.path.join(result_dir, f'{fpath_sans_ext}.json')
        results = {}

        # Run each algorithm 
        predictions = {}
        for name, algo in algorithm_dict.items():
            # Run algorithm
            pred_onsets, pred, onset_function, seconds = algo(gt_onsets,fpath, out_dir)
            
            if name != "Double_threshold_onset":
               visualize_onsets(gt_onsets, pred_onsets , seconds, onset_function, fname, plot_dir, algo_name=name)
            # Evaluate algorithm
            result = evaluate_algorithm(gt, pred)
            predictions[name] = result
        # Store results
        results[fpath] = predictions

        # Save results to json file
        with open(json_file, 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    main()