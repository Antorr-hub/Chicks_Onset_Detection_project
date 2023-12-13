# from mir_eval_modified import util_mod
from mir_eval_modified import onset
import evaluation as myeval
import os
import pandas as pd
import numpy as np
import json

import onset_detection_algorithms as onset_detectors


# audiofile = "C:\\Users\\anton\Data_experiment\\Data\\Training_set\\chick41_d0.wav"
audiofile = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav"
save_evaluation_results_path = r'./example_results/'
# predictions_in_seconds = onset_detectors.high_frequency_content(audiofile)      #using the default parameters! because they are defined in the function we do not need to write them here!

# evaluate
load_predictions_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/example_results/chick41_d0_HFCpredictions.csv'
predictions_in_seconds = pd.read_csv(load_predictions_path)['onset_seconds'].values
# get ground truth onsets
gt_onsets = myeval.get_reference_onsets(audiofile.replace('.wav', '.txt'))

Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, predictions_in_seconds, window=0.05)

# save Lists of TP, FP, FN for visualisation in sonic visualiser
evaluation_results = { 'audiofilename' : os.path.basename(audiofile), 'Algorithm':'HFC',  'f_score':Fscore, 'precision':precision, 'recall': recall}
TP_pd = pd.DataFrame(TP, columns=['TP'])
TP_pd.to_csv(os.path.join(save_evaluation_results_path, os.path.basename(load_predictions_path[:-4]) +'_TP.csv'), index=False)
FP_pd = pd.DataFrame(FP, columns=['FP'])
FP_pd.to_csv(os.path.join(save_evaluation_results_path, os.path.basename(load_predictions_path[:-4]) +'_FP.csv'), index=False)
FN_pd = pd.DataFrame(FN, columns=['FN'])
FN_pd.to_csv(os.path.join(save_evaluation_results_path, os.path.basename(load_predictions_path[:-4]) +'_FN.csv'), index=False)
with open(os.path.join(save_evaluation_results_path, os.path.basename(load_predictions_path[:-4]) +'_evaluation_results.json'), 'w') as fp:
    json.dump(evaluation_results, fp)






print('stop')