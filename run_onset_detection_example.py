
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import onset_detection_algorithms as onset_detectors
import mir_eval_modified as mir_eval_new


# from mir_eval_modified.onset import f_measure
import evaluation as eval
from visualization import visualize_activation_and_gt




audiofile = "C:\\Users\\anton\\High_quality_dataset\\chick87_d0.wav"
metadata = pd.read_csv("C:\\Users\\anton\\High_quality_dataset\\high_quality_dataset_metadata.csv")
#audiofile = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav"

save_predictions_path = r'./example_results/'
if not os.path.exists(save_predictions_path):
    os.mkdir(save_predictions_path)
    

# HFCpredictions_in_seconds, activation_frames = onset_detectors.high_frequency_content(audiofile, visualise_activation=True)      #using the default parameters! because they are defined in the function we do not need to write them here!

# TPDpredictions_in_seconds = onset_detectors.thresholded_phase_deviation(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

# NWPDpredictions_in_seconds = onset_detectors.normalized_weighted_phase_deviation(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

# RCDpredictions_in_seconds = onset_detectors.rectified_complex_domain(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

Spf_predictions_in_seconds, activation_in_frames, hop_length, sr_superflux  = onset_detectors.superflux(audiofile, visualise_activation=True) #using the default parameters! because they are defined in the function we do not need to write them here!

# Dbt_predictions_in_seconds = onset_detectors.double_threshold(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!
# save prediction to file
# predictions_seconds_df = pd.DataFrame(predictions_in_seconds, columns=['onset_seconds'])
# predictions_seconds_df.to_csv(os.path.join(save_predictions_path, os.path.basename(audiofile[:-4]) +'_HFCpredictions.csv'), index=False)
# print(f"predictions_in_seconds: {HFCpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {TPDpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {NWPDpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {RCDpredictions_in_seconds[:10]}")

print(f"predictions_in_seconds: {Spf_predictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {Dbt_predictions_in_seconds[:10]}")


# ##evaluate
# get ground truth onsets
# gt_onsets = eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))
# exp_start = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['Start_experiment_sec'].values[0]   
# exp_end = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['End_experiment_sec'].values[0]
# gt_onsets, HFCpredictions_in_seconds, activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
#                                                 gt_onsets, HFCpredictions_in_seconds, activation_frames)
    
# compute individual scores Fmeasure, precision, recall 
# scores_hfc = mir_eval.onset.evaluate(gt_onsets, HFCpredictions_in_seconds, window=0.1)

# scores_tpd = mir_eval.onset.evaluate(gt_onsets, TPDpredictions_in_seconds, window=0.05)

# scores_nwpd = mir_eval.onset.evaluate(gt_onsets, NWPDpredictions_in_seconds, window=0.05)

# scores_rcd = mir_eval.onset.evaluate(gt_onsets, RCDpredictions_in_seconds, window=0.05)


gt_onsets = eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))
exp_start = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['Start_experiment_sec'].values[0]   
exp_end = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['End_experiment_sec'].values[0]
gt_onsets, spfpredictions_in_seconds, activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                gt_onsets, Spf_predictions_in_seconds, activation_in_frames, hop_length, sr_superflux)

fmeasure, precision, recall,TP, FP, FN  = mir_eval_new.onset.f_measure(gt_onsets, spfpredictions_in_seconds, window=0.5)

# scores_dbt = mir_eval.onset.evaluate(gt_onsets, Dbt_predictions_in_seconds, window=0.05)




# print(f"Scores: {scores_hfc}")

# print(f"Scores: {scores_tpd}")

# print(f"Scores: {scores_nwpd}")

# print(f"Scores: {scores_rcd}")

# print(f"Scores: {scores_spf}")

# print(f"Scores: {scores_dbt}")

# visualise

visualize_activation_and_gt(plot_dir=save_predictions_path,file_name=os.path.basename(audiofile), onset_detection_funtion_name='superflux',
                             gt_onsets=gt_onsets, activation=activation_frames, start_exp=exp_start, end_exp=exp_end ,hop_length=hop_length, sr=sr_superflux)



print('done')








