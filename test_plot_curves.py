import os
import onset_detection_algorithms as onset_detectors
import evaluation as eval
import mir_eval_modified as mir_eval_mod
import numpy as np
from mir_eval_modified import onset
import visualization as vis




thresholds = np.arange(0.02, 1.3, 0.1)
eval_window = 0.1

# file_name = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav'



# gt_onsets = gt_onsets = eval.get_reference_onsets(file_name.replace('.wav', '.txt'))

# precisions = []
# recalls = [] 

# for th in thresholds:


#     predictions_scnd = onset_detectors.thresholded_phase_deviation(file_name, hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000,
#                                                                 spec_fref=2500, spec_alpha= 0.95,pp_threshold=th, pp_pre_avg=0, pp_post_avg=0, 
#                                                                 pp_pre_max=10, pp_post_max=10)
#      # TODO: modify parameters of TPD with the best ones given by the grid search

#     _, prec, rec, _,_,_ = onset.f_measure(gt_onsets, predictions_scnd, window=eval_window)

#     precisions.append(prec)
#     recalls.append(rec)

data_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised'
onset_detector_function = 'thresholded_phase_deviation'
dataset = 'train_val_normalised'
precisions, recalls = eval.compute_precision_recall_curve(onset_detectors.thresholded_phase_deviation,data_folder, thresholds, eval_window=0.1)

vis.plot_precision_recall_thresholds(thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png')
vis.plot_precision_recall_curve(precisions, recalls)
           


