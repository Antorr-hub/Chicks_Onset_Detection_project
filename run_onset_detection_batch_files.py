<<<<<<< HEAD
import os
import glob
import onset_detection_algorithms as onset_detectors
from tqdm import tqdm
import pandas as pd
import evaluation as my_eval
from mir_eval_modified.onset import f_measure
import json
from visualization import visualize_activation_and_gt


def run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder):
    """
    :param onset_detector_function: function that takes as input an audio file and returns a list of onsets in seconds
    :param onset_detector_parameter_dict: dictionary with the parameters of the onset_detector_function
    :param dataset_csv: csv file with the metadata of the dataset
    :param dataset_folder: folder with the audio files of the dataset
    :return: saves the results of the onset detection algorithm to a folder
    """

    individual_fscore_list = []
    individual_precision_list = []
    individual_recall_list = []
    n_events_list = []

    metadata = pd.read_csv(dataset_csv)
    list_files = glob.glob(os.path.join(dataset_folder, "*.wav"))

    for audiofile in tqdm(list_files):
        
        # create folder for each chick results
        chick_folder = os.path.join(save_results_folder, os.path.basename(audiofile[:-4]))
        if not os.path.exists(chick_folder):
            os.mkdir(chick_folder)

        # get ground truth onsets
        gt_onsets = my_eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))
        n_events_list.append(len(gt_onsets))
        # discard events outside experiment window
        exp_start = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['Start_experiment_sec'].values[0]
        exp_end = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['End_experiment_sec'].values[0]

        chick = os.path.basename(audiofile)[:-4]

        # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
        if onset_detector_function.__name__ == 'superflux':
            pred_scnd, activation_frames, spec_hop_length, spec_sr = onset_detector_function(audiofile, visualise_activation=True, **onset_detector_parameter_dict) #superflux returns 4 things instead of 3 if visualise=true, also names of parameters are different!
        else:
            pred_scnd, activation_frames = onset_detector_function(audiofile, visualise_activation=True, **onset_detector_parameter_dict)

        # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
        if onset_detector_function.__name__ == 'superflux':
            gt_onsets, pred_scnd, activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
                                                                                                        gt_onsets,
                                                                                                        pred_scnd,
                                                                                                        activation_frames,
                                                                                                        hop_length=spec_hop_length,
                                                                                                        sr=spec_sr)
        else:
            gt_onsets, pred_scnd, activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
                                                                                                        gt_onsets,
                                                                                                        pred_scnd,
                                                                                                        activation_frames,
                                                                                                        hop_length=onset_detector_parameter_dict['hop_length'],
                                                                                                        sr=onset_detector_parameter_dict['sr'])
        
        # shift predictions a constant offset:
        pred_scnd = my_eval.global_shift_correction(pred_scnd, evaluation_parameters_dict['correction_shift'])

        # correct double onsets
        pred_scnd = my_eval.double_onset_correction(pred_scnd, evaluation_parameters_dict['double_onset_correction'])

        # scores_hfc = mir_eval.onset.evaluate(gt_onsets, hfc_pred_scnd, window=evaluation_window)
        Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, pred_scnd, window=evaluation_parameters_dict['eval_window'])

        individual_fscore_list.append(Fscore)
        individual_precision_list.append(precision)
        individual_recall_list.append(recall)

        # save Lists of TP, FP, FN for visualisation in sonic visualiser
        evaluation_results = {'audiofilename': chick, 'Algorithm': onset_detector_function.__name__, 'F-measure': Fscore,
                                'Precision': precision, 'Recall': recall}
        TP_pd = pd.DataFrame(TP, columns=['TP'])
        TP_pd.to_csv(os.path.join(chick_folder, f'{chick}_TP.csv'), index=False)
        FP_pd = pd.DataFrame(FP, columns=['FP'])
        FP_pd.to_csv(os.path.join(chick_folder, f'{chick}_FP.csv'), index=False)
        FN_pd = pd.DataFrame(FN, columns=['FN'])
        FN_pd.to_csv(os.path.join(chick_folder, f'{chick}_FN.csv'), index=False)

        with open(os.path.join(chick_folder, f"{chick}_{onset_detector_function.__name__}_evaluation_results.json"), 'w') as fp:
            json.dump(evaluation_results, fp)
        if onset_detector_function.__name__ == 'superflux':
            visualize_activation_and_gt(plot_dir=chick_folder, file_name=os.path.basename(audiofile),
                                    onset_detection_funtion_name=onset_detector_function.__name__,
                                    gt_onsets=gt_onsets, activation=activation_frames, start_exp=exp_start,
                                    end_exp=exp_end, hop_length=spec_hop_length,
                                    sr=spec_sr)
        else: 
            visualize_activation_and_gt(plot_dir=chick_folder, file_name=os.path.basename(audiofile),
                                    onset_detection_funtion_name=onset_detector_function.__name__,
                                    gt_onsets=gt_onsets, activation=activation_frames, start_exp=exp_start,
                                    end_exp=exp_end, hop_length=onset_detector_parameter_dict['hop_length'],
                                    sr=onset_detector_parameter_dict['sr'])
        # save prediction to file
        predictions_seconds_df = pd.DataFrame(pred_scnd, columns=['onset_seconds'])
        predictions_seconds_df.to_csv(os.path.join(chick_folder, chick + f'_{onset_detector_function.__name__}_predictions.csv'), index=False)


    # compute weighted average
    # print(
    global_fmeasure_in_batch =  my_eval.compute_weighted_average(individual_fscore_list, n_events_list)
    global_precision_in_batch = my_eval.compute_weighted_average(individual_precision_list, n_events_list)
    global_recall_in_batch = my_eval.compute_weighted_average(individual_recall_list, n_events_list)

    # save global results to file
    global_results = {'dataset': os.path.basename(dataset_folder), 'Algorithm': onset_detector_function.__name__, 'F-measure': global_fmeasure_in_batch,
                        'Precision': global_precision_in_batch, 'Recall': global_recall_in_batch}
    
    with open(os.path.join(save_results_folder, f"{os.path.basename(dataset_folder)}_{onset_detector_function.__name__}_global_results.json"), 'w') as fp:
        json.dump(global_results, fp)

    return print(f'Global results for {onset_detector_function.__name__} on {os.path.basename(dataset_folder)} dataset: F-measure: {global_fmeasure_in_batch}, Precision: {global_precision_in_batch}, Recall: {global_recall_in_batch}')

    

if __name__ == '__main__':
    dataset_csv = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set/chicks_testing_metadata.csv'
    dataset_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set'

    onset_detector_function = onset_detectors.high_frequency_content
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'spec_num_bands': 15, 'spec_fmin': 2500,   
                                    'spec_fmax': 5000, 'spec_fref': 2800, 'pp_threshold': 1.8, 'pp_pre_avg': 25,
                                    'pp_post_avg': 1, 'pp_pre_max': 3, 'pp_post_max': 2}
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.05, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/HFC_with_best_parameters_from_grid_search_on_test_set_and_window01_with_DOC'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    onset_detector_function = onset_detectors.high_frequency_content
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'spec_num_bands': 15, 'spec_fmin': 2500,   
                                    'spec_fmax': 5000, 'spec_fref': 2800, 'pp_threshold': 1.8, 'pp_pre_avg': 25,
                                    'pp_post_avg': 1, 'pp_pre_max': 3, 'pp_post_max': 2}
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.15, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/HFC_with_best_parameters_from_grid_search_on_test_set_with_window_0.5_with_DOC'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)



    onset_detector_function = onset_detectors.superflux
    onset_detector_parameter_dict = {'spec_hop_length': 1024 // 2, 'spec_n_fft' : 2048 * 2, 'spec_window': 0.12,
                                    'spec_n_mels': 15, 'spec_fmin': 2050, 'spec_fmax': 8000, 'spec_lag': 3 , 'spec_max_size':60, 
                                    'pp_pre_avg':10, 'pp_post_avg':10, 'pp_pre_max':1, 'pp_post_max':1, 'pp_threshold':0.1, 'pp_wait':10}
    
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.05, 'double_onset_correction': 0.1}

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Superflux_with_best_parameters_from_grid_search_on_test_set_and_01window'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)

    onset_detector_parameter_dict = {'spec_hop_length': 1024 // 2, 'spec_n_fft' : 2048 * 2, 'spec_window': 0.12,
                                    'spec_n_mels': 15, 'spec_fmin': 2050, 'spec_fmax': 8000, 'spec_lag': 3 , 'spec_max_size':60, 
                                    'pp_pre_avg':10, 'pp_post_avg':10, 'pp_pre_max':1, 'pp_post_max':1, 'pp_threshold':0.1, 'pp_wait':10}
    
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.05, 'double_onset_correction': 0.1}

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Superflux_with_best_parameters_from_grid_search_on_test_set'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)



    onset_detector_function = onset_detectors.rectified_complex_domain
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'pp_threshold': 70, 'pp_pre_avg': 20,
                                    'pp_post_avg': 20, 'pp_pre_max': 10, 'pp_post_max': 10}
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.1, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/RCD_with_best_parameters_from_grid_search_on_test_set'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)

    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'pp_threshold': 70, 'pp_pre_avg': 20,
                                    'pp_post_avg': 20, 'pp_pre_max': 10, 'pp_post_max': 10}
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.1, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/RCD_with_best_parameters_from_grid_search_on_test_set_and_window01'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)


# #  WHEN running with parameters different from deafult, change here and name the results folder with the new parameters
# # EVAL_WINDOW = 0.1

# # DEFAULT PARAMETERS FOR EACH ALGORITHM  
# # TODO modify calling these functions to take parameters as input  done for HFC
# # onset_detector_parameter_dict = {}

# # HFC:   hop_length=441, sr=44100, spec_num_bands=12, spec_fmin=1800, spec_fmax=6500, 
# #        spec_fref=2500, pp_threshold= 2.5, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=1, pp_post_max=1, visualise_activation=False
# HFC_parameters = {'hop_length': 441, 'sr':44100, 'spec_num_bands':15, 'spec_fmin': 2500, 'spec_fmax': 5000, 'spec_fref' : 2800,
#                   'pp_threshold':  1, 'pp_pre_avg' :3, 'pp_post_avg':0, 'pp_pre_max':3, 'pp_post_max':2, 'eval_window':0.5, 'correction_shift':0.02}

# # TPD:   hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000,spec_fref=2500,spec_alpha= 0.95,
#  #       pp_threshold=0.95, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10, visualise_activation=False

# # TPD_parameters= {'hop_length':441, 'sr':44100, 'spec_num_bands':64, 'spec_fmin':1800, 'spec_fmax':6000,'spec_fref':2500,'spec_alpha': 0.95,
# #         'pp_threshold':0.95, 'pp_pre_avg':0, 'pp_post_avg':0, 'pp_pre_max':10, 'pp_post_max':10}


# # # NWPD:  hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000, spec_fref=2500, 
# # #        pp_threshold=0.1, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10

# # NWPD_parameters= {'hop_length': 441, 'sr':44100, 'pp_threshold': 0.92, 'pp_pre_avg':0, 'pp_post_avg':0, 'pp_pre_max':30, 'pp_post_max':30}



# # # RCD:   hop_length=441, sr=44100
# # #        pp_threshold=50, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=10, pp_post_max=10  
# # RCD_parameters= {'hop_length':441, 'sr':44100, 'pp_threshold': 50, 'pp_pre_avg':25, 'pp_post_avg':25, 'pp_pre_max':10, 'pp_post_max':10}

# # # Superflux:  spec_hop_length=1024 // 2, spec_n_fft=2048 *2, spec_window=0.12, spec_fmin=2050, spec_fmax=6000,
# # #                         spec_n_mels=15, spec_lag=5, spec_max_size=50, visualise_activation=False, pp_threshold=0)

# # Superflux_parameters= {'hop_length': 1024 // 2, 'n_fft': 2048 * 2, 'window': 0.12, 'fmin': 2050, 'fmax': 6000,
# #                          'n_mels': 15, 'lag': 5, 'max_size': 50, 'visualise_activation': False, 'pp_threshold': 0}
                       
# # # DBT:   sr= 44100, hop_length=441, spec_n_fft=2048, spec_window=0.12

# # DBT_parameters= {'sr': 44100, 'hop_length': 441, 'n_fft': 2048, 'window': 0.12}

# # ###############################
# audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set'
# # audio_folder = 'C:\\Users\\anton\\Data_normalised\\Testing_set'


# # metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
# metadata = pd.read_csv("/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set/chicks_testing_metadata.csv")

# # save_evaluation_results_path = r'C:\Users\anton\Chicks_Onset_Detection_project\Testing_norm_dataset_default_params_results'
# save_evaluation_results_path = r'/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Best_parameters_from_grid_search'
# #######################

# if not os.path.exists(save_evaluation_results_path):
#     os.makedirs(save_evaluation_results_path)



# n_events_list = []
# list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

# individual_fscore_list_HFC = []
# individual_precision_list_HFC = []
# individual_recall_list_HFC = []


# individual_fscore_list_TPD = []
# individual_precision_list_TPD = []
# individual_recall_list_TPD = []


# individual_fscore_list_NWPD = []
# individual_precision_list_NWPD = []
# individual_recall_list_NWPD = []


# individual_fscore_list_RCD = []
# individual_precision_list_RCD = []
# individual_recall_list_RCD = []


# individual_fscore_list_Superflux = []
# individual_precision_list_Superflux = []
# individual_recall_list_Superflux = []


# individual_fscore_list_DBT = []
# individual_precision_list_DBT = []
# individual_recall_list_DBT = []





# for file in tqdm(list_files):

#     # create folder for eaach chick results
#     chick_folder = os.path.join(save_evaluation_results_path, os.path.basename(file[:-4]))
#     if not os.path.exists(chick_folder):
#         os.mkdir(chick_folder)

#     # get ground truth onsets
#     gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
#     # discard events outside experiment window
#     exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]   
#     exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    
#     chick = os.path.basename(file)[:-4]
    



#     #  High Frequency Content   
 
#     HFC_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'HFC')
#     if not os.path.exists(HFC_results_folder):
#         os.mkdir(HFC_results_folder)
    
#     # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
#     hfc_pred_scnd,  HFC_activation_frames = onset_detectors.high_frequency_content(file, hop_length=HFC_parameters['hop_length'], sr=HFC_parameters['sr'], 
#                                                                                     spec_num_bands=HFC_parameters['spec_num_bands'], spec_fmin=HFC_parameters['spec_fmin'],
#                                                                                     spec_fmax=HFC_parameters['spec_fmax'], spec_fref=HFC_parameters['spec_fref'],
#                                                                                     pp_threshold= HFC_parameters['pp_threshold'], pp_pre_avg=HFC_parameters['pp_pre_avg'],
#                                                                                     pp_post_avg=HFC_parameters['pp_post_avg'], pp_pre_max=HFC_parameters['pp_pre_max'],
#                                                                                     pp_post_max=HFC_parameters['pp_post_max'], visualise_activation=True)

#     # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
#     gt_onsets, hfc_pred_scnd, HFC_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end, gt_onsets, hfc_pred_scnd, 
#                                                                                                        HFC_activation_frames , hop_length=HFC_parameters['hop_length'], 
#                                                                                                        sr=HFC_parameters['sr'])
    

#     # shift predictions a constant offset:
#     hfc_pred_scnd = my_eval.global_shift_correction(hfc_pred_scnd, HFC_parameters['correction_shift'] )
    
#     #scores_hfc = mir_eval.onset.evaluate(gt_onsets, hfc_pred_scnd, window=evaluation_window)
#     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, hfc_pred_scnd , window= HFC_parameters['eval_window'])

#     individual_fscore_list_HFC.append(Fscore)
#     individual_precision_list_HFC.append(precision)
#     individual_recall_list_HFC.append(recall)
#     # save Lists of TP, FP, FN for visualisation in sonic visualiser
#     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'High frequency content',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
#     TP_pd = pd.DataFrame(TP, columns=['TP'])
#     TP_pd.to_csv(os.path.join(HFC_results_folder,  f'{chick}_TP.csv'), index=False)
#     FP_pd = pd.DataFrame(FP, columns=['FP'])
#     FP_pd.to_csv(os.path.join(HFC_results_folder, f'{chick}_FP.csv'), index=False)
#     FN_pd = pd.DataFrame(FN, columns=['FN'])
#     FN_pd.to_csv(os.path.join(HFC_results_folder,  f'{chick}_FN.csv'), index=False)

#     with open(os.path.join(HFC_results_folder,  f"{chick}_HFC_evaluation_results.json"), 'w') as fp:
#         json.dump(evaluation_results, fp)

#     visualize_activation_and_gt(plot_dir=HFC_results_folder,file_name=os.path.basename(file),onset_detection_funtion_name='High frequency content',
#                              gt_onsets=gt_onsets, activation= HFC_activation_frames, start_exp=exp_start, end_exp=exp_end ,hop_length=HFC_parameters['hop_length'], sr=HFC_parameters['sr'])
#  # save prediction to file
#     hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])
#     hfc_predictions_seconds_df.to_csv(os.path.join(HFC_results_folder, chick +'_HFCpredictions.csv'), index=False)




# # ###############Evaluation for Thresholded Phase Deviation

# #     TPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'TPD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(TPD_results_folder):
# #         os.mkdir(TPD_results_folder)

# #     tpd_pred_scnd, TPD_activation_frames = onset_detectors.thresholded_phase_deviation(file, visualise_activation=True)



# #     # get ground truth onsets, TPDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, tpd_pred_scnd, TPD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
# #                                                     gt_onsets, tpd_pred_scnd, TPD_activation_frames, hop_length=441, sr=44100)




# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, tpd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_TPD.append(Fscore)
# #     individual_precision_list_TPD.append(precision)
# #     individual_recall_list_TPD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Thresholded Phase Deviation',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(TPD_results_folder,f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(TPD_results_folder,f"_{chick}_TPD_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=TPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Thresholded Phase Deviation', 
# #                                              gt_onsets=gt_onsets, activation= TPD_activation_frames ,start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    
    

    
# #     tpd_predictions_seconds_df = pd.DataFrame(tpd_pred_scnd, columns=['onset_seconds'])
# #     tpd_predictions_seconds_df.to_csv(os.path.join(TPD_results_folder, chick +'_TPDpredictions.csv'), index=False)





# # ###############Evaluation for Normalized Weighted Phase Deviation

# #     NWPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'NWPD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(NWPD_results_folder):
# #         os.mkdir(NWPD_results_folder)

# #     nwpd_pred_scnd, NWPD_activation_frames = onset_detectors.normalized_weighted_phase_deviation(file, visualise_activation=True)



# #     # get ground truth onsets, NWPDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, nwpd_pred_scnd, NWPD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                     gt_onsets, nwpd_pred_scnd, NWPD_activation_frames,hop_length=441, sr=44100 )
    

# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, nwpd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_NWPD.append(Fscore)
# #     individual_precision_list_NWPD.append(precision)
# #     individual_recall_list_NWPD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Normalised Weighted Phase Deviation',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(NWPD_results_folder,f"_{chick}_NWPD_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=NWPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Normalised Weighted Phase Deviation', 
# #                                              gt_onsets=gt_onsets, activation= NWPD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    



    
# #     nwpd_pred_scnd_df = pd.DataFrame(nwpd_pred_scnd, columns=['onset_seconds'])
# #     nwpd_pred_scnd_df.to_csv(os.path.join(NWPD_results_folder, chick +'_NWPDpredictions.csv'), index=False)




# # ###############Evaluation for Rectified Complex Domain
# #     RCD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'RCD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(RCD_results_folder):
# #         os.mkdir(RCD_results_folder)

# #     rcd_pred_scnd, RCD_activation_frames = onset_detectors.rectified_complex_domain(file, visualise_activation=True)
  
# #     # get ground truth onsets, RCDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets,  rcd_pred_scnd, RCD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                                                 gt_onsets, rcd_pred_scnd, RCD_activation_frames, hop_length=441, sr=44100)
    
    

# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, rcd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_RCD.append(Fscore)
# #     individual_precision_list_RCD.append(precision)
# #     individual_recall_list_RCD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Rectified Complex Domain',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(RCD_results_folder,f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(RCD_results_folder, f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=RCD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Rectified Complex Domain', 
# #                                             gt_onsets=gt_onsets, activation= RCD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)

    

# #     rcd_pred_scnd_df = pd.DataFrame(rcd_pred_scnd, columns=['onset_seconds'])
# #     rcd_pred_scnd_df.to_csv(os.path.join(RCD_results_folder, chick +'_RCDpredictions.csv'), index=False)



# # ###############Evaluation for Superflux
# #     Superflux_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'Superflux_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(Superflux_results_folder):
# #         os.mkdir(Superflux_results_folder)


# #     spf_pred_scnd, Superflux_activation_frames, spf_hop, spf_sr = onset_detectors.superflux(file, visualise_activation=True)

# #     # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, spf_pred_scnd, Superflux_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                                                 gt_onsets, spf_pred_scnd, Superflux_activation_frames, hop_length= spf_hop , sr= spf_sr)


# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, spf_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_Superflux.append(Fscore)
# #     individual_precision_list_Superflux.append(precision)
# #     individual_recall_list_Superflux.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Superflux',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(Superflux_results_folder,f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=Superflux_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Superflux', 
# #                                         gt_onsets=gt_onsets, activation= Superflux_activation_frames,start_exp=exp_start, end_exp=exp_end, hop_length= spf_hop , sr= spf_sr)    
    
 


# #     spf_pred_scnd_df = pd.DataFrame(spf_pred_scnd, columns=['onset_seconds']) 
# #     spf_pred_scnd_df.to_csv(os.path.join(Superflux_results_folder, chick +'_SPFpredictions.csv'), index=False)


# # ####Evaluation for Double Threshold
# #     DBT_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'DBT_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(DBT_results_folder):
# #         os.mkdir(DBT_results_folder)
# #     dbt_pred_scnd = onset_detectors.double_threshold(file)
# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, dbt_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_DBT.append(Fscore)
# #     individual_precision_list_DBT.append(precision)
# #     individual_recall_list_DBT.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Double Threshold',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(DBT_results_folder , f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(DBT_results_folder , f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)


# #     dbt_pred_scnd_df = pd.DataFrame(dbt_pred_scnd, columns=['onset_seconds'])
# #     dbt_pred_scnd_df.to_csv(os.path.join(DBT_results_folder, chick +'_DBTpredictions.csv'), index=False)



#     n_events_list.append(len(gt_onsets))


# # compute weighted average
# # print(
# global_fmeasure_HFC_in_batch =  my_eval.compute_weighted_average(individual_fscore_list_HFC, n_events_list)
# global_precision_HFC_in_batch = my_eval.compute_weighted_average(individual_precision_list_HFC, n_events_list)
# global_recall_JFC_in_batch = my_eval.compute_weighted_average(individual_recall_list_HFC, n_events_list)

# # # compute weighted average
# # global_f1score_TPD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_TPD, n_events_list)
# # global_precision_TPD_in_batch = my_eval.compute_weighted_average(individual_precision_list_TPD, n_events_list)
# # global_recall_TPD_in_batch = my_eval.compute_weighted_average(individual_recall_list_TPD, n_events_list)

# # # compute weighted average
# # global_f1score_NWPD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_NWPD, n_events_list)
# # global_precision_NWPD_in_batch = my_eval.compute_weighted_average(individual_precision_list_NWPD, n_events_list)
# # global_recall_NWPD_in_batch = my_eval.compute_weighted_average(individual_recall_list_NWPD, n_events_list)


# # # compute weighted average
# # global_f1score_RCD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_RCD, n_events_list)
# # global_precision_RCD_in_batch = my_eval.compute_weighted_average(individual_precision_list_RCD, n_events_list)
# # global_recall_RCD_in_batch = my_eval.compute_weighted_average(individual_recall_list_RCD, n_events_list)

# # # compute weighted average
# # global_f1score_Superflux_in_batch = my_eval.compute_weighted_average(individual_fscore_list_Superflux, n_events_list)
# # global_precision_Superflux_in_batch = my_eval.compute_weighted_average(individual_precision_list_Superflux, n_events_list)
# # global_recall_Superflux_in_batch = my_eval.compute_weighted_average(individual_recall_list_Superflux, n_events_list)

# # global_f1score_DBT_in_batch = my_eval.compute_weighted_average(individual_fscore_list_DBT, n_events_list)
# # global_precision_DBT_in_batch = my_eval.compute_weighted_average(individual_precision_list_DBT, n_events_list)
# # global_recall_DBT_in_batch = my_eval.compute_weighted_average(individual_recall_list_DBT, n_events_list)


# globals_results_dict = { 'HFC': {'F1-measure': global_fmeasure_HFC_in_batch, 'Precision': global_precision_HFC_in_batch, 'Recall': global_recall_JFC_in_batch, 'parameters': HFC_parameters},
#                         # 'TPD': {'F1-measure': global_f1score_TPD_in_batch, 'Precision': global_precision_TPD_in_batch, 'Recall': global_recall_TPD_in_batch, 'parameters': TPD_parameters},
#                         # 'NWPD': {'F1-measure': global_f1score_NWPD_in_batch, 'Precision': global_precision_NWPD_in_batch, 'Recall': global_recall_NWPD_in_batch, 'parameters': NWPD_parameters},
#                         # 'RCD': {'F1-measure': global_f1score_RCD_in_batch, 'Precision': global_precision_RCD_in_batch, 'Recall': global_recall_RCD_in_batch, 'parameters': RCD_parameters},
#                         # 'Superflux': {'F1-measure': global_f1score_Superflux_in_batch, 'Precision': global_precision_Superflux_in_batch, 'Recall': global_recall_Superflux_in_batch, 'parameters': Superflux_parameters},
#                         # 'DBT': {'F1-measure': global_f1score_DBT_in_batch, 'Precision': global_precision_DBT_in_batch, 'Recall': global_recall_DBT_in_batch, 'parameters': DBT_parameters}
#                         }
# with open(os.path.join(save_evaluation_results_path, "global_evaluation_results.json"), 'w') as fp:
#     json.dump(globals_results_dict, fp)

# print('done')



=======
import os
import glob
import onset_detection_algorithms as onset_detectors
from tqdm import tqdm
import pandas as pd
import evaluation as my_eval
from mir_eval_modified.onset import f_measure
import json
from visualization import visualize_activation_and_gt


def run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder):
    """
    :param onset_detector_function: function that takes as input an audio file and returns a list of onsets in seconds
    :param onset_detector_parameter_dict: dictionary with the parameters of the onset_detector_function
    :param dataset_csv: csv file with the metadata of the dataset
    :param dataset_folder: folder with the audio files of the dataset
    :return: saves the results of the onset detection algorithm to a folder
    """

    individual_fscore_list = []
    individual_precision_list = []
    individual_recall_list = []
    n_events_list = []

    metadata = pd.read_csv(dataset_csv)
    list_files = glob.glob(os.path.join(dataset_folder, "*.wav"))

    for audiofile in tqdm(list_files):
        
        # create folder for each chick results
        chick_folder = os.path.join(save_results_folder, os.path.basename(audiofile[:-4]))
        if not os.path.exists(chick_folder):
            os.mkdir(chick_folder)

        # get ground truth onsets
        gt_onsets = my_eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))
        n_events_list.append(len(gt_onsets))
        # discard events outside experiment window
        exp_start = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['Start_experiment_sec'].values[0]
        exp_end = metadata[metadata['Filename'] == os.path.basename(audiofile)[:-4]]['End_experiment_sec'].values[0]

        chick = os.path.basename(audiofile)[:-4]

        # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
        if onset_detector_function.__name__ == 'superflux':
            pred_scnd, activation_frames, spec_hop_length, spec_sr = onset_detector_function(audiofile, visualise_activation=True, **onset_detector_parameter_dict) #superflux returns 4 things instead of 3 if visualise=true, also names of parameters are different!
        else:
            pred_scnd, activation_frames = onset_detector_function(audiofile, visualise_activation=True, **onset_detector_parameter_dict)

        # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
        if onset_detector_function.__name__ == 'superflux':
            gt_onsets, pred_scnd, activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
                                                                                                        gt_onsets,
                                                                                                        pred_scnd,
                                                                                                        activation_frames,
                                                                                                        hop_length=spec_hop_length,
                                                                                                        sr=spec_sr)
        else:
            gt_onsets, pred_scnd, activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
                                                                                                        gt_onsets,
                                                                                                        pred_scnd,
                                                                                                        activation_frames,
                                                                                                        hop_length=onset_detector_parameter_dict['hop_length'],
                                                                                                        sr=onset_detector_parameter_dict['sr'])
        
        # shift predictions a constant offset:
        pred_scnd = my_eval.global_shift_correction(pred_scnd, evaluation_parameters_dict['correction_shift'])

        # correct double onsets
        pred_scnd = my_eval.double_onset_correction(pred_scnd, evaluation_parameters_dict['double_onset_correction'])

        # scores_hfc = mir_eval.onset.evaluate(gt_onsets, hfc_pred_scnd, window=evaluation_window)
        Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, pred_scnd, window=evaluation_parameters_dict['eval_window'])

        individual_fscore_list.append(Fscore)
        individual_precision_list.append(precision)
        individual_recall_list.append(recall)

        # save Lists of TP, FP, FN for visualisation in sonic visualiser
        evaluation_results = {'audiofilename': chick, 'Algorithm': onset_detector_function.__name__, 'F-measure': Fscore,
                                'Precision': precision, 'Recall': recall}
        TP_pd = pd.DataFrame(TP, columns=['TP'])
        TP_pd.to_csv(os.path.join(chick_folder, f'{chick}_TP.csv'), index=False)
        FP_pd = pd.DataFrame(FP, columns=['FP'])
        FP_pd.to_csv(os.path.join(chick_folder, f'{chick}_FP.csv'), index=False)
        FN_pd = pd.DataFrame(FN, columns=['FN'])
        FN_pd.to_csv(os.path.join(chick_folder, f'{chick}_FN.csv'), index=False)

        with open(os.path.join(chick_folder, f"{chick}_{onset_detector_function.__name__}_evaluation_results.json"), 'w') as fp:
            json.dump(evaluation_results, fp)
        if onset_detector_function.__name__ == 'superflux':
            visualize_activation_and_gt(plot_dir=chick_folder, file_name=os.path.basename(audiofile),
                                    onset_detection_funtion_name=onset_detector_function.__name__,
                                    gt_onsets=gt_onsets, activation=activation_frames, start_exp=exp_start,
                                    end_exp=exp_end, hop_length=spec_hop_length,
                                    sr=spec_sr)
        else: 
            visualize_activation_and_gt(plot_dir=chick_folder, file_name=os.path.basename(audiofile),
                                    onset_detection_funtion_name=onset_detector_function.__name__,
                                    gt_onsets=gt_onsets, activation=activation_frames, start_exp=exp_start,
                                    end_exp=exp_end, hop_length=onset_detector_parameter_dict['hop_length'],
                                    sr=onset_detector_parameter_dict['sr'])
        # save prediction to file
        predictions_seconds_df = pd.DataFrame(pred_scnd, columns=['onset_seconds'])
        predictions_seconds_df.to_csv(os.path.join(chick_folder, chick + f'_{onset_detector_function.__name__}_predictions.csv'), index=False)


    # compute weighted average
    # print(
    global_fmeasure_in_batch =  my_eval.compute_weighted_average(individual_fscore_list, n_events_list)
    global_precision_in_batch = my_eval.compute_weighted_average(individual_precision_list, n_events_list)
    global_recall_in_batch = my_eval.compute_weighted_average(individual_recall_list, n_events_list)

    # save global results to file
    global_results = {'dataset': os.path.basename(dataset_folder), 'Algorithm': onset_detector_function.__name__, 'F-measure': global_fmeasure_in_batch,
                        'Precision': global_precision_in_batch, 'Recall': global_recall_in_batch}
    
    with open(os.path.join(save_results_folder, f"{os.path.basename(dataset_folder)}_{onset_detector_function.__name__}_global_results.json"), 'w') as fp:
        json.dump(global_results, fp)

    return print(f'Global results for {onset_detector_function.__name__} on {os.path.basename(dataset_folder)} dataset: F-measure: {global_fmeasure_in_batch}, Precision: {global_precision_in_batch}, Recall: {global_recall_in_batch}')

    

if __name__ == '__main__':
    dataset_csv = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set/chicks_testing_metadata.csv'
    dataset_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set'

    onset_detector_function = onset_detectors.high_frequency_content
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'spec_num_bands': 15, 'spec_fmin': 2500,   
                                    'spec_fmax': 5000, 'spec_fref': 2800, 'pp_threshold': 1.8, 'pp_pre_avg': 25,
                                    'pp_post_avg': 1, 'pp_pre_max': 3, 'pp_post_max': 2}
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.05, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/HFC_with_best_parameters_from_grid_search_on_test_set_and_window01_with_DOC'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    onset_detector_function = onset_detectors.high_frequency_content
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'spec_num_bands': 15, 'spec_fmin': 2500,   
                                    'spec_fmax': 5000, 'spec_fref': 2800, 'pp_threshold': 1.8, 'pp_pre_avg': 25,
                                    'pp_post_avg': 1, 'pp_pre_max': 3, 'pp_post_max': 2}
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.15, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/HFC_with_best_parameters_from_grid_search_on_test_set_with_window_0.5_with_DOC'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)



    onset_detector_function = onset_detectors.superflux
    onset_detector_parameter_dict = {'spec_hop_length': 1024 // 2, 'spec_n_fft' : 2048 * 2, 'spec_window': 0.12,
                                    'spec_n_mels': 15, 'spec_fmin': 2050, 'spec_fmax': 8000, 'spec_lag': 3 , 'spec_max_size':60, 
                                    'pp_pre_avg':10, 'pp_post_avg':10, 'pp_pre_max':1, 'pp_post_max':1, 'pp_threshold':0.1, 'pp_wait':10}
    
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.05, 'double_onset_correction': 0.1}

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Superflux_with_best_parameters_from_grid_search_on_test_set_and_01window'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)

    onset_detector_parameter_dict = {'spec_hop_length': 1024 // 2, 'spec_n_fft' : 2048 * 2, 'spec_window': 0.12,
                                    'spec_n_mels': 15, 'spec_fmin': 2050, 'spec_fmax': 8000, 'spec_lag': 3 , 'spec_max_size':60, 
                                    'pp_pre_avg':10, 'pp_post_avg':10, 'pp_pre_max':1, 'pp_post_max':1, 'pp_threshold':0.1, 'pp_wait':10}
    
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.05, 'double_onset_correction': 0.1}

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Superflux_with_best_parameters_from_grid_search_on_test_set'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)



    onset_detector_function = onset_detectors.rectified_complex_domain
    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'pp_threshold': 70, 'pp_pre_avg': 20,
                                    'pp_post_avg': 20, 'pp_pre_max': 10, 'pp_post_max': 10}
    evaluation_parameters_dict = {'eval_window': 0.5, 'correction_shift': 0.1, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/RCD_with_best_parameters_from_grid_search_on_test_set'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)

    onset_detector_parameter_dict = {'hop_length': 441, 'sr': 44100, 'pp_threshold': 70, 'pp_pre_avg': 20,
                                    'pp_post_avg': 20, 'pp_pre_max': 10, 'pp_post_max': 10}
    evaluation_parameters_dict = {'eval_window': 0.1, 'correction_shift': 0.1, 'double_onset_correction': 0.1}
    

    save_results_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/RCD_with_best_parameters_from_grid_search_on_test_set_and_window01'
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    run_onset_detector_on_batch_files(onset_detector_function, onset_detector_parameter_dict, evaluation_parameters_dict, dataset_csv, dataset_folder, save_results_folder)


# #  WHEN running with parameters different from deafult, change here and name the results folder with the new parameters
# # EVAL_WINDOW = 0.1

# # DEFAULT PARAMETERS FOR EACH ALGORITHM  
# # TODO modify calling these functions to take parameters as input  done for HFC
# # onset_detector_parameter_dict = {}

# # HFC:   hop_length=441, sr=44100, spec_num_bands=12, spec_fmin=1800, spec_fmax=6500, 
# #        spec_fref=2500, pp_threshold= 2.5, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=1, pp_post_max=1, visualise_activation=False
# HFC_parameters = {'hop_length': 441, 'sr':44100, 'spec_num_bands':15, 'spec_fmin': 2500, 'spec_fmax': 5000, 'spec_fref' : 2800,
#                   'pp_threshold':  1, 'pp_pre_avg' :3, 'pp_post_avg':0, 'pp_pre_max':3, 'pp_post_max':2, 'eval_window':0.5, 'correction_shift':0.02}

# # TPD:   hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000,spec_fref=2500,spec_alpha= 0.95,
#  #       pp_threshold=0.95, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10, visualise_activation=False

# # TPD_parameters= {'hop_length':441, 'sr':44100, 'spec_num_bands':64, 'spec_fmin':1800, 'spec_fmax':6000,'spec_fref':2500,'spec_alpha': 0.95,
# #         'pp_threshold':0.95, 'pp_pre_avg':0, 'pp_post_avg':0, 'pp_pre_max':10, 'pp_post_max':10}


# # # NWPD:  hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000, spec_fref=2500, 
# # #        pp_threshold=0.1, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10

# # NWPD_parameters= {'hop_length': 441, 'sr':44100, 'pp_threshold': 0.92, 'pp_pre_avg':0, 'pp_post_avg':0, 'pp_pre_max':30, 'pp_post_max':30}



# # # RCD:   hop_length=441, sr=44100
# # #        pp_threshold=50, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=10, pp_post_max=10  
# # RCD_parameters= {'hop_length':441, 'sr':44100, 'pp_threshold': 50, 'pp_pre_avg':25, 'pp_post_avg':25, 'pp_pre_max':10, 'pp_post_max':10}

# # # Superflux:  spec_hop_length=1024 // 2, spec_n_fft=2048 *2, spec_window=0.12, spec_fmin=2050, spec_fmax=6000,
# # #                         spec_n_mels=15, spec_lag=5, spec_max_size=50, visualise_activation=False, pp_threshold=0)

# # Superflux_parameters= {'hop_length': 1024 // 2, 'n_fft': 2048 * 2, 'window': 0.12, 'fmin': 2050, 'fmax': 6000,
# #                          'n_mels': 15, 'lag': 5, 'max_size': 50, 'visualise_activation': False, 'pp_threshold': 0}
                       
# # # DBT:   sr= 44100, hop_length=441, spec_n_fft=2048, spec_window=0.12

# # DBT_parameters= {'sr': 44100, 'hop_length': 441, 'n_fft': 2048, 'window': 0.12}

# # ###############################
# audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set'
# # audio_folder = 'C:\\Users\\anton\\Data_normalised\\Testing_set'


# # metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Testing_set\\chicks_testing_metadata.csv")
# metadata = pd.read_csv("/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/data/Testing_set/chicks_testing_metadata.csv")

# # save_evaluation_results_path = r'C:\Users\anton\Chicks_Onset_Detection_project\Testing_norm_dataset_default_params_results'
# save_evaluation_results_path = r'/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Best_parameters_from_grid_search'
# #######################

# if not os.path.exists(save_evaluation_results_path):
#     os.makedirs(save_evaluation_results_path)



# n_events_list = []
# list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

# individual_fscore_list_HFC = []
# individual_precision_list_HFC = []
# individual_recall_list_HFC = []


# individual_fscore_list_TPD = []
# individual_precision_list_TPD = []
# individual_recall_list_TPD = []


# individual_fscore_list_NWPD = []
# individual_precision_list_NWPD = []
# individual_recall_list_NWPD = []


# individual_fscore_list_RCD = []
# individual_precision_list_RCD = []
# individual_recall_list_RCD = []


# individual_fscore_list_Superflux = []
# individual_precision_list_Superflux = []
# individual_recall_list_Superflux = []


# individual_fscore_list_DBT = []
# individual_precision_list_DBT = []
# individual_recall_list_DBT = []





# for file in tqdm(list_files):

#     # create folder for eaach chick results
#     chick_folder = os.path.join(save_evaluation_results_path, os.path.basename(file[:-4]))
#     if not os.path.exists(chick_folder):
#         os.mkdir(chick_folder)

#     # get ground truth onsets
#     gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
#     # discard events outside experiment window
#     exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]   
#     exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    
#     chick = os.path.basename(file)[:-4]
    



#     #  High Frequency Content   
 
#     HFC_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'HFC')
#     if not os.path.exists(HFC_results_folder):
#         os.mkdir(HFC_results_folder)
    
#     # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
#     hfc_pred_scnd,  HFC_activation_frames = onset_detectors.high_frequency_content(file, hop_length=HFC_parameters['hop_length'], sr=HFC_parameters['sr'], 
#                                                                                     spec_num_bands=HFC_parameters['spec_num_bands'], spec_fmin=HFC_parameters['spec_fmin'],
#                                                                                     spec_fmax=HFC_parameters['spec_fmax'], spec_fref=HFC_parameters['spec_fref'],
#                                                                                     pp_threshold= HFC_parameters['pp_threshold'], pp_pre_avg=HFC_parameters['pp_pre_avg'],
#                                                                                     pp_post_avg=HFC_parameters['pp_post_avg'], pp_pre_max=HFC_parameters['pp_pre_max'],
#                                                                                     pp_post_max=HFC_parameters['pp_post_max'], visualise_activation=True)

#     # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
#     gt_onsets, hfc_pred_scnd, HFC_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end, gt_onsets, hfc_pred_scnd, 
#                                                                                                        HFC_activation_frames , hop_length=HFC_parameters['hop_length'], 
#                                                                                                        sr=HFC_parameters['sr'])
    

#     # shift predictions a constant offset:
#     hfc_pred_scnd = my_eval.global_shift_correction(hfc_pred_scnd, HFC_parameters['correction_shift'] )
    
#     #scores_hfc = mir_eval.onset.evaluate(gt_onsets, hfc_pred_scnd, window=evaluation_window)
#     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, hfc_pred_scnd , window= HFC_parameters['eval_window'])

#     individual_fscore_list_HFC.append(Fscore)
#     individual_precision_list_HFC.append(precision)
#     individual_recall_list_HFC.append(recall)
#     # save Lists of TP, FP, FN for visualisation in sonic visualiser
#     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'High frequency content',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
#     TP_pd = pd.DataFrame(TP, columns=['TP'])
#     TP_pd.to_csv(os.path.join(HFC_results_folder,  f'{chick}_TP.csv'), index=False)
#     FP_pd = pd.DataFrame(FP, columns=['FP'])
#     FP_pd.to_csv(os.path.join(HFC_results_folder, f'{chick}_FP.csv'), index=False)
#     FN_pd = pd.DataFrame(FN, columns=['FN'])
#     FN_pd.to_csv(os.path.join(HFC_results_folder,  f'{chick}_FN.csv'), index=False)

#     with open(os.path.join(HFC_results_folder,  f"{chick}_HFC_evaluation_results.json"), 'w') as fp:
#         json.dump(evaluation_results, fp)

#     visualize_activation_and_gt(plot_dir=HFC_results_folder,file_name=os.path.basename(file),onset_detection_funtion_name='High frequency content',
#                              gt_onsets=gt_onsets, activation= HFC_activation_frames, start_exp=exp_start, end_exp=exp_end ,hop_length=HFC_parameters['hop_length'], sr=HFC_parameters['sr'])
#  # save prediction to file
#     hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])
#     hfc_predictions_seconds_df.to_csv(os.path.join(HFC_results_folder, chick +'_HFCpredictions.csv'), index=False)




# # ###############Evaluation for Thresholded Phase Deviation

# #     TPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'TPD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(TPD_results_folder):
# #         os.mkdir(TPD_results_folder)

# #     tpd_pred_scnd, TPD_activation_frames = onset_detectors.thresholded_phase_deviation(file, visualise_activation=True)



# #     # get ground truth onsets, TPDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, tpd_pred_scnd, TPD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start, exp_end,
# #                                                     gt_onsets, tpd_pred_scnd, TPD_activation_frames, hop_length=441, sr=44100)




# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, tpd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_TPD.append(Fscore)
# #     individual_precision_list_TPD.append(precision)
# #     individual_recall_list_TPD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Thresholded Phase Deviation',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(TPD_results_folder,f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(TPD_results_folder,f"_{chick}_TPD_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=TPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Thresholded Phase Deviation', 
# #                                              gt_onsets=gt_onsets, activation= TPD_activation_frames ,start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    
    

    
# #     tpd_predictions_seconds_df = pd.DataFrame(tpd_pred_scnd, columns=['onset_seconds'])
# #     tpd_predictions_seconds_df.to_csv(os.path.join(TPD_results_folder, chick +'_TPDpredictions.csv'), index=False)





# # ###############Evaluation for Normalized Weighted Phase Deviation

# #     NWPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'NWPD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(NWPD_results_folder):
# #         os.mkdir(NWPD_results_folder)

# #     nwpd_pred_scnd, NWPD_activation_frames = onset_detectors.normalized_weighted_phase_deviation(file, visualise_activation=True)



# #     # get ground truth onsets, NWPDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, nwpd_pred_scnd, NWPD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                     gt_onsets, nwpd_pred_scnd, NWPD_activation_frames,hop_length=441, sr=44100 )
    

# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, nwpd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_NWPD.append(Fscore)
# #     individual_precision_list_NWPD.append(precision)
# #     individual_recall_list_NWPD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Normalised Weighted Phase Deviation',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(NWPD_results_folder,f"_{chick}_NWPD_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=NWPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Normalised Weighted Phase Deviation', 
# #                                              gt_onsets=gt_onsets, activation= NWPD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    



    
# #     nwpd_pred_scnd_df = pd.DataFrame(nwpd_pred_scnd, columns=['onset_seconds'])
# #     nwpd_pred_scnd_df.to_csv(os.path.join(NWPD_results_folder, chick +'_NWPDpredictions.csv'), index=False)




# # ###############Evaluation for Rectified Complex Domain
# #     RCD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'RCD_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(RCD_results_folder):
# #         os.mkdir(RCD_results_folder)

# #     rcd_pred_scnd, RCD_activation_frames = onset_detectors.rectified_complex_domain(file, visualise_activation=True)
  
# #     # get ground truth onsets, RCDpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets,  rcd_pred_scnd, RCD_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                                                 gt_onsets, rcd_pred_scnd, RCD_activation_frames, hop_length=441, sr=44100)
    
    

# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, rcd_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_RCD.append(Fscore)
# #     individual_precision_list_RCD.append(precision)
# #     individual_recall_list_RCD.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Rectified Complex Domain',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(RCD_results_folder,f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(RCD_results_folder, f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=RCD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Rectified Complex Domain', 
# #                                             gt_onsets=gt_onsets, activation= RCD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)

    

# #     rcd_pred_scnd_df = pd.DataFrame(rcd_pred_scnd, columns=['onset_seconds'])
# #     rcd_pred_scnd_df.to_csv(os.path.join(RCD_results_folder, chick +'_RCDpredictions.csv'), index=False)



# # ###############Evaluation for Superflux
# #     Superflux_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'Superflux_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(Superflux_results_folder):
# #         os.mkdir(Superflux_results_folder)


# #     spf_pred_scnd, Superflux_activation_frames, spf_hop, spf_sr = onset_detectors.superflux(file, visualise_activation=True)

# #     # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
# #     gt_onsets, spf_pred_scnd, Superflux_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
# #                                                                                 gt_onsets, spf_pred_scnd, Superflux_activation_frames, hop_length= spf_hop , sr= spf_sr)


# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, spf_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_Superflux.append(Fscore)
# #     individual_precision_list_Superflux.append(precision)
# #     individual_recall_list_Superflux.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Superflux',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(Superflux_results_folder,f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)

# #     visualize_activation_and_gt(plot_dir=Superflux_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Superflux', 
# #                                         gt_onsets=gt_onsets, activation= Superflux_activation_frames,start_exp=exp_start, end_exp=exp_end, hop_length= spf_hop , sr= spf_sr)    
    
 


# #     spf_pred_scnd_df = pd.DataFrame(spf_pred_scnd, columns=['onset_seconds']) 
# #     spf_pred_scnd_df.to_csv(os.path.join(Superflux_results_folder, chick +'_SPFpredictions.csv'), index=False)


# # ####Evaluation for Double Threshold
# #     DBT_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'DBT_default_parameters_window' + str(EVAL_WINDOW))
# #     if not os.path.exists(DBT_results_folder):
# #         os.mkdir(DBT_results_folder)
# #     dbt_pred_scnd = onset_detectors.double_threshold(file)
# #     Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, dbt_pred_scnd, window=EVAL_WINDOW)
# #     individual_fscore_list_DBT.append(Fscore)
# #     individual_precision_list_DBT.append(precision)
# #     individual_recall_list_DBT.append(recall)
# #     # save Lists of TP, FP, FN for visualisation in sonic visualiser
# #     evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Double Threshold',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
# #     TP_pd = pd.DataFrame(TP, columns=['TP'])
# #     TP_pd.to_csv(os.path.join(DBT_results_folder , f'_{chick}_TP.csv'), index=False)
# #     FP_pd = pd.DataFrame(FP, columns=['FP'])
# #     FP_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FP.csv'), index=False)
# #     FN_pd = pd.DataFrame(FN, columns=['FN'])
# #     FN_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FN.csv'), index=False)
# #     with open(os.path.join(DBT_results_folder , f"_{chick}_evaluation_results.json"), 'w') as fp:
# #         json.dump(evaluation_results, fp)


# #     dbt_pred_scnd_df = pd.DataFrame(dbt_pred_scnd, columns=['onset_seconds'])
# #     dbt_pred_scnd_df.to_csv(os.path.join(DBT_results_folder, chick +'_DBTpredictions.csv'), index=False)



#     n_events_list.append(len(gt_onsets))


# # compute weighted average
# # print(
# global_fmeasure_HFC_in_batch =  my_eval.compute_weighted_average(individual_fscore_list_HFC, n_events_list)
# global_precision_HFC_in_batch = my_eval.compute_weighted_average(individual_precision_list_HFC, n_events_list)
# global_recall_JFC_in_batch = my_eval.compute_weighted_average(individual_recall_list_HFC, n_events_list)

# # # compute weighted average
# # global_f1score_TPD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_TPD, n_events_list)
# # global_precision_TPD_in_batch = my_eval.compute_weighted_average(individual_precision_list_TPD, n_events_list)
# # global_recall_TPD_in_batch = my_eval.compute_weighted_average(individual_recall_list_TPD, n_events_list)

# # # compute weighted average
# # global_f1score_NWPD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_NWPD, n_events_list)
# # global_precision_NWPD_in_batch = my_eval.compute_weighted_average(individual_precision_list_NWPD, n_events_list)
# # global_recall_NWPD_in_batch = my_eval.compute_weighted_average(individual_recall_list_NWPD, n_events_list)


# # # compute weighted average
# # global_f1score_RCD_in_batch = my_eval.compute_weighted_average(individual_fscore_list_RCD, n_events_list)
# # global_precision_RCD_in_batch = my_eval.compute_weighted_average(individual_precision_list_RCD, n_events_list)
# # global_recall_RCD_in_batch = my_eval.compute_weighted_average(individual_recall_list_RCD, n_events_list)

# # # compute weighted average
# # global_f1score_Superflux_in_batch = my_eval.compute_weighted_average(individual_fscore_list_Superflux, n_events_list)
# # global_precision_Superflux_in_batch = my_eval.compute_weighted_average(individual_precision_list_Superflux, n_events_list)
# # global_recall_Superflux_in_batch = my_eval.compute_weighted_average(individual_recall_list_Superflux, n_events_list)

# # global_f1score_DBT_in_batch = my_eval.compute_weighted_average(individual_fscore_list_DBT, n_events_list)
# # global_precision_DBT_in_batch = my_eval.compute_weighted_average(individual_precision_list_DBT, n_events_list)
# # global_recall_DBT_in_batch = my_eval.compute_weighted_average(individual_recall_list_DBT, n_events_list)


# globals_results_dict = { 'HFC': {'F1-measure': global_fmeasure_HFC_in_batch, 'Precision': global_precision_HFC_in_batch, 'Recall': global_recall_JFC_in_batch, 'parameters': HFC_parameters},
#                         # 'TPD': {'F1-measure': global_f1score_TPD_in_batch, 'Precision': global_precision_TPD_in_batch, 'Recall': global_recall_TPD_in_batch, 'parameters': TPD_parameters},
#                         # 'NWPD': {'F1-measure': global_f1score_NWPD_in_batch, 'Precision': global_precision_NWPD_in_batch, 'Recall': global_recall_NWPD_in_batch, 'parameters': NWPD_parameters},
#                         # 'RCD': {'F1-measure': global_f1score_RCD_in_batch, 'Precision': global_precision_RCD_in_batch, 'Recall': global_recall_RCD_in_batch, 'parameters': RCD_parameters},
#                         # 'Superflux': {'F1-measure': global_f1score_Superflux_in_batch, 'Precision': global_precision_Superflux_in_batch, 'Recall': global_recall_Superflux_in_batch, 'parameters': Superflux_parameters},
#                         # 'DBT': {'F1-measure': global_f1score_DBT_in_batch, 'Precision': global_precision_DBT_in_batch, 'Recall': global_recall_DBT_in_batch, 'parameters': DBT_parameters}
#                         }
# with open(os.path.join(save_evaluation_results_path, "global_evaluation_results.json"), 'w') as fp:
#     json.dump(globals_results_dict, fp)

# print('done')



>>>>>>> edc2ece3053e14d0a1981b46a2a0ac0c14f776fe
