import os
import onset_detection_algorithms as onset_detectors
import evaluation as eval
import mir_eval_modified as mir_eval_mod
import numpy as np





thresholds = np.arange(0.02, 0.1, 1.3)
file_name = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav'
onset_detectors.thresholded_phase_deviation(file_name, hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000, 
                   spec_fref=2500, spec_alpha= 0.95,pp_threshold= [0.95], pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10)

