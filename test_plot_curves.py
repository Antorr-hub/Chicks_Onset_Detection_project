import os
import onset_detection_algorithms as onset_detectors
import evaluation as my_eval
import mir_eval_modified as mir_eval_mod
import numpy as np
from mir_eval_modified import onset
import visualization as vis
import pandas as pd
from tqdm import tqdm
import glob


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

data_folder = 'C:\\Users\\anton\\Data_normalised\\Testing_set'


#call the metadata
metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")


save_evaluation_results_path = r'C:\Users\anton\Chicks_Onset_Detection_project\Precision_recall_curves_testing_set_pre_tuning'
if not os.path.exists(save_evaluation_results_path):
    os.mkdir(save_evaluation_results_path)
    #Precision_recall_curves_testing_set_pre_tuning


list_files = glob.glob(os.path.join(data_folder, "*.wav"))

for file in tqdm(list_files):

    # get ground truth onsets
    gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    # discard events outside experiment window
    exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]   
    exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    
    chick = os.path.basename(file)[:-4]
    dataset = 'Testing_set'


#    # High Frequency Content 
    # peak picking parameters tested in the grid search
    # threshold_range = [1.8, 2.5, 3, 3.5]  
    # best threshold = 1.8  with f-measure = 0.79

Hfc_thresholds = np.linspace(0.02, 5, 100)  # for the grid search we have tested 1.8, 2.5, 3
onset_detector_function = 'High_Frequency_Content'
precisions, recalls = my_eval.compute_precision_recall_curve(onset_detectors.high_frequency_content, data_folder, Hfc_thresholds, exp_start, exp_end, eval_window= eval_window)


vis.plot_precision_recall_thresholds(Hfc_thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png')
vis.plot_precision_recall_curve(precisions, recalls, save_file_name='Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png')
# add to the forlder 
vis.plot_precision_recall_thresholds(Hfc_thresholds, precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png'))
vis.plot_precision_recall_curve(precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png'))          


Rcd_thresholds = np.linspace(20, 150, 100) # for the grid search we have tested 30, 50, 70
onset_detector_function = 'Rectified_Complex_Domain'

precisions, recalls = my_eval.compute_precision_recall_curve(onset_detectors.rectified_complex_domain, data_folder, Rcd_thresholds, exp_start, exp_end,  eval_window= eval_window)

vis.plot_precision_recall_thresholds(Rcd_thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+ onset_detector_function+'_'+dataset+'.png')
vis.plot_precision_recall_curve(precisions, recalls, save_file_name='Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png')

vis.plot_precision_recall_thresholds(Rcd_thresholds, precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png'))
# add to the forlde
vis.plot_precision_recall_curve(precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png'))          
vis.plot_precision_recall_thresholds(Rcd_thresholds, precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png'))


# Superflux
# grid search parameters tested
# delta_range = delta_range = [0, 0.03, 0.05, 0.08, 0.1, 0.2]
# best delta = 0.03 with f-measure = 0.838 on val and 0.738 on test


# Superflux_thresholds = np.linspace(0, 0.3, 100) # for the grid search we have tested delta= 0
# onset_detector_function = 'Superflux'
# precisions, recalls = my_eval.compute_precision_recall_curve(onset_detectors.superflux, data_folder, Superflux_thresholds, exp_start, exp_end, eval_window= eval_window, hop_length= 512, sr= 44100)

# vis.plot_precision_recall_thresholds(Superflux_thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+ onset_detector_function+'_'+dataset+'.png')
# vis.plot_precision_recall_curve(precisions, recalls, save_file_name='Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png')
# vis.plot_precision_recall_thresholds(Superflux_thresholds, precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_vs_thresholds_curve_'+onset_detector_function+'_'+dataset+'.png'))
# vis.plot_precision_recall_curve(precisions, recalls, save_file_name=os.path.join(save_evaluation_results_path, 'Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png'))           




# Tpd_thresholds = np.linspace(0.87, 1.2, 5) # for the grid search we have tested 0.9, 0.95
# onset_detector_function = 'Thresholded_Phase_Deviation'
# precisions, recalls = my_eval.compute_precision_recall_curve(onset_detectors.thresholded_phase_deviation, data_folder, Tpd_thresholds, exp_start, exp_end,  eval_window= eval_window)


# vis.plot_precision_recall_thresholds(Tpd_thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+ onset_detector_function+'_'+dataset+'.png')
# vis.plot_precision_recall_curve(precisions, recalls, save_file_name='Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png')






# Nwpd_thresholds = np.linspace(0.65, 1.5, 5) #for the grid search we have tested 0.8, 0.92, 0.95

# onset_detector_function = 'Normalized_Weighted_Phase_Deviation'
# precisions, recalls = my_eval.compute_precision_recall_curve(onset_detectors.normalized_weighted_phase_deviation, data_folder, Nwpd_thresholds, exp_start, exp_end, eval_window= eval_window)

# vis.plot_precision_recall_thresholds(Nwpd_thresholds, precisions, recalls, save_file_name='Precision_recall_vs_thresholds_curve_'+ onset_detector_function+'_'+dataset+'.png')
# vis.plot_precision_recall_curve(precisions, recalls, save_file_name='Precision_recall_curve_'+onset_detector_function+'_'+dataset+'.png')







