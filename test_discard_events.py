# from mir_eval_modified import util_mod
from mir_eval_modified import onset
import evaluation as my_eval
import os
import pandas as pd
import numpy as np
import json


import onset_detection_algorithms as onset_detectors


EVAL_WINDOW = 0.1

# audiofile = "C:\\Users\\anton\Data_experiment\\Data\\Training_set\\chick41_d0.wav"
audiofile = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick21_d0.wav"
save_evaluation_results_path = r'./example_results/'
metadata = pd.read_csv("/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/chicks_training_metadata.csv")

Spf_predictions_in_seconds_all, activation_in_frames, hop_length, sr_superflux  = onset_detectors.superflux(audiofile, spec_hop_length=1024 // 2, spec_n_fft=2048 *2, spec_window=0.12, spec_fmin=2050, spec_fmax=6000,
                         spec_n_mels=15, spec_lag=5, spec_max_size=50, visualise_activation=True, pp_threshold=0) #using the default parameters! because they are defined in the function we do not need to write them here!

print(f"predictions_in_seconds: {Spf_predictions_in_seconds_all[:10]}")
print('len predictions_in_seconds: ', len(Spf_predictions_in_seconds_all))

gt_onsets_all = my_eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))
print('len ground_truth_onsets: ', len(gt_onsets_all))



exp_start = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['Start_experiment_sec'].values[0]   
exp_end = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['End_experiment_sec'].values[0]


fmeasure, prec, recall, TP, FP, FN =onset.f_measure(gt_onsets_all, Spf_predictions_in_seconds_all, window=EVAL_WINDOW)

print('fmeasure: ', fmeasure)
print('precision: ', prec)
print('recall: ', recall)
print('len(TP): ', len(TP))
print('len(FP): ', len(FP))
print('len(FN): ', len(FN))

gt_onsets, spfpredictions_in_seconds, activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                gt_onsets_all, Spf_predictions_in_seconds_all, activation_in_frames, hop_length, sr_superflux)

print('len modified predictions_in_seconds: ', len(spfpredictions_in_seconds))
print('len modified ground_truth_onsets: ', len(gt_onsets))


fmeasure_exp_window, precision_exp_window, recall_exp_window,TP_exp_wind, FP_exp_wind, FN_exp_wind = onset.f_measure(gt_onsets, spfpredictions_in_seconds, window=EVAL_WINDOW)

print('fmeasure_exp_window: ', fmeasure_exp_window)
print('precision_exp_window: ', precision_exp_window)
print('recall_exp_window: ', recall_exp_window)
print('len(TP_exp_wind): ', len(TP_exp_wind))
print('FP')
print('len(FP_exp_wind): ', len(FP_exp_wind))
print('len(FN_exp_wind): ', len(FN_exp_wind))





print('stop')

