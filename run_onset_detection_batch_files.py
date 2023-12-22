import os
import glob
import onset_detection_algorithms as onset_detectors
from tqdm import tqdm
import pandas as pd
import mir_eval
#import mir_eval_modified as onset
from mir_eval_modified.onset import f_measure

#import mir_eval_modified as mir_eval_new


import evaluation as eval
import json
from visualization import visualize_activation_and_gt



#audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/'
audio_folder = 'C:\\Users\\anton\\High_quality_dataset'
#save_predictions_path = './Results_normalised_data_default_parameters/'

metadata = pd.read_csv("C:\\Users\\anton\\High_quality_dataset\\high_quality_dataset_metadata.csv")

save_evaluation_results_path = r'C:\Users\anton\Chicks_Onset_Detection_project\High_quality_dataset_results'
if not os.path.exists(save_evaluation_results_path):
    os.mkdir(save_evaluation_results_path)
    

evaluation_window = 0.5
n_events_list = []
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

individual_fscore_list_HFC = []
individual_precision_list_HFC = []
individual_recall_list_HFC = []


individual_fscore_list_TPD = []
individual_precision_list_TPD = []
individual_recall_list_TPD = []


individual_fscore_list_NWPD = []
individual_precision_list_NWPD = []
individual_recall_list_NWPD = []


individual_fscore_list_RCD = []
individual_precision_list_RCD = []
individual_recall_list_RCD = []


individual_fscore_list_Superflux = []
individual_precision_list_Superflux = []
individual_recall_list_Superflux = []


individual_fscore_list_DBT = []
individual_precision_list_DBT = []
individual_recall_list_DBT = []





for file in tqdm(list_files):

    # create folder for eaach chick results
    chick_folder = os.path.join(save_evaluation_results_path, os.path.basename(file[:-4]))
    if not os.path.exists(chick_folder):
        os.mkdir(chick_folder)

    # get ground truth onsets
    gt_onsets = eval.get_reference_onsets(file.replace('.wav', '.txt'))
    # discard events outside experiment window
    exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]   
    exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    
    chick = os.path.basename(file)[:-4]
    
 ###############Evaluation for High Frequency Content   
    # for alg in list_algorithms:
    HFC_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'HFC_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(HFC_results_folder):
        os.mkdir(HFC_results_folder)

    # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
    hfc_pred_scnd,  HFC_activation_frames = onset_detectors.high_frequency_content(file, visualise_activation=True)

    ''' The onset detection function for High frequency content take as arguments:

    the file name
    visualise_activation= True if you want to visualise the activation function

    Returns:

    predicted onsets in seconds (list) 
    frames of the function (list)

    '''
    
    # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, HFCpredictions_in_seconds, HFC_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                    gt_onsets, hfc_pred_scnd, HFC_activation_frames , hop_length=441, sr=44100)
    

    '''
    To test only the grund thruth onsets and the predicted onsets sampled inside the experiment window, 
    The function take as arguments:
    
    exp_start=start of the experiment in seconds( milliseconds in decimals)
    exp_end= end of the experiment in seconds ( millisecons in decimals)
    gt_onsets= ground thruth onsets
    hf_pred_scnd= onsets precicted in seconds
    HFC_activation_frames= the activation function in time frames.

    Returns:
    gt_onsets= ground thruth onsets inside the experiment window
    hf_pred_scnd= onsets precicted in seconds inside the experiment window
    HFC_activation_frames= the activation function in time frames inside the experiment window.
    
    '''
    
    # compute individual scores Fmeasure, precision, recall

    #scores_hfc = mir_eval.onset.evaluate(gt_onsets, hfc_pred_scnd, window=evaluation_window)
    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, hfc_pred_scnd, window= evaluation_window)
    '''
    To compute the F-measure, precision, recall, TP, FP, FN, the function take as arguments:
    gt_onsets= ground thruth onsets inside the experiment window
    hf_pred_scnd= onsets precicted in seconds inside the experiment window
    window= evaluation window in seconds
    '''


    individual_fscore_list_HFC.append(Fscore)
    individual_precision_list_HFC.append(precision)
    individual_recall_list_HFC.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : chick , 'Algorithm':'High frequency content',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(HFC_results_folder,  f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(HFC_results_folder, f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(HFC_results_folder,  f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(HFC_results_folder,  f"_{chick}_HFC_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=HFC_results_folder,file_name=os.path.basename(file),onset_detection_funtion_name='High frequency content',
                             gt_onsets=gt_onsets, activation= HFC_activation_frames, start_exp=exp_start, end_exp=exp_end ,hop_length=441, sr=44100)
    
    '''
    To visualise the activation function and the ground thruth onsets, the function take as arguments:
    plot_dir= path to the folder where you want to save the plot
    file_name= name of the file
    onset_detection_funtion_name= name of the onset detection function
    gt_onsets= ground thruth onsets inside the experiment window
    activation= the activation function in time frames inside the experiment window.
    start_exp= start of the experiment in seconds( seconds in decimals)
    end_exp= end of the experiment in seconds ( seconDs in decimals)
    hop_length= hop length in samples
    sr= sampling rate in Hz
    '''
    

 # save prediction to file
    hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])
    hfc_predictions_seconds_df.to_csv(os.path.join(HFC_results_folder, chick +'_HFCpredictions.csv'), index=False)




###############Evaluation for Thresholded Phase Deviation

    TPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'TPD_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(TPD_results_folder):
        os.mkdir(TPD_results_folder)

    tpd_pred_scnd, TPD_activation_frames = onset_detectors.thresholded_phase_deviation(file, visualise_activation=True)



    # get ground truth onsets, TPDpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, TPDpredictions_in_seconds, TPD_activation_frames = eval.discard_events_outside_experiment_window(exp_start, exp_end,
                                                    gt_onsets, tpd_pred_scnd, TPD_activation_frames, hop_length=441, sr=44100)




    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, tpd_pred_scnd, window=evaluation_window)
    individual_fscore_list_TPD.append(Fscore)
    individual_precision_list_TPD.append(precision)
    individual_recall_list_TPD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Thresholded Phase Deviation',  'F_measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(TPD_results_folder, f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(TPD_results_folder,f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(TPD_results_folder,f"_{chick}_TPD_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=TPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Thresholded Phase Deviation', 
                                             gt_onsets=gt_onsets, activation= TPD_activation_frames ,start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    
    

    
    tpd_predictions_seconds_df = pd.DataFrame(tpd_pred_scnd, columns=['onset_seconds'])
    tpd_predictions_seconds_df.to_csv(os.path.join(TPD_results_folder, chick +'_TPDpredictions.csv'), index=False)





###############Evaluation for Normalized Weighted Phase Deviation

    NWPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'NWPD_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(NWPD_results_folder):
        os.mkdir(NWPD_results_folder)

    nwpd_pred_scnd, NWPD_activation_frames = onset_detectors.normalized_weighted_phase_deviation(file, visualise_activation=True)



    # get ground truth onsets, NWPDpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, NWPDpredictions_in_seconds, NWPD_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end,
                                                    gt_onsets, nwpd_pred_scnd, NWPD_activation_frames,hop_length=441, sr=44100 )
    

    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, nwpd_pred_scnd, window=evaluation_window)
    individual_fscore_list_NWPD.append(Fscore)
    individual_precision_list_NWPD.append(precision)
    individual_recall_list_NWPD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Normalised Weighted Phase Deviation',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(NWPD_results_folder, f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(NWPD_results_folder,f"_{chick}_NWPD_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=NWPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Normalised Weighted Phase Deviation', 
                                             gt_onsets=gt_onsets, activation= NWPD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)    



    
    nwpd_pred_scnd_df = pd.DataFrame(nwpd_pred_scnd, columns=['onset_seconds'])
    nwpd_pred_scnd_df.to_csv(os.path.join(NWPD_results_folder, chick +'_NWPDpredictions.csv'), index=False)




###############Evaluation for Rectified Complex Domain
    RCD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'RCD_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(RCD_results_folder):
        os.mkdir(RCD_results_folder)

    rcd_pred_scnd, RCD_activation_frames = onset_detectors.rectified_complex_domain(file, visualise_activation=True)
  
    # get ground truth onsets, RCDpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, RCDpredictions_in_seconds, RCD_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end,
                                                                                gt_onsets, rcd_pred_scnd, RCD_activation_frames, hop_length=441, sr=44100)
    
    

    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, rcd_pred_scnd, window=evaluation_window)
    individual_fscore_list_RCD.append(Fscore)
    individual_precision_list_RCD.append(precision)
    individual_recall_list_RCD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Rectified Complex Domain',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(RCD_results_folder,f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(RCD_results_folder, f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(RCD_results_folder, f"_{chick}_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=RCD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Rectified Complex Domain', 
                                            gt_onsets=gt_onsets, activation= RCD_activation_frames, start_exp=exp_start, end_exp=exp_end, hop_length=441, sr=44100)

    

    rcd_pred_scnd_df = pd.DataFrame(rcd_pred_scnd, columns=['onset_seconds'])
    rcd_pred_scnd_df.to_csv(os.path.join(RCD_results_folder, chick +'_RCDpredictions.csv'), index=False)



###############Evaluation for Superflux
    Superflux_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'Superflux_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(Superflux_results_folder):
        os.mkdir(Superflux_results_folder)


    spf_pred_scnd, Superflux_activation_frames, spf_hop, spf_sr = onset_detectors.superflux(file, visualise_activation=True)

    # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, Superfluxpredictions_in_seconds, Superflux_activation_frames = eval.discard_events_outside_experiment_window(exp_start,exp_end,
                                                                                gt_onsets, spf_pred_scnd, Superflux_activation_frames, hop_length= spf_hop , sr= spf_sr)


    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, spf_pred_scnd, window=evaluation_window)
    individual_fscore_list_Superflux.append(Fscore)
    individual_precision_list_Superflux.append(precision)
    individual_recall_list_Superflux.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Superflux',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(Superflux_results_folder, f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(Superflux_results_folder,f"_{chick}_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=Superflux_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Superflux', 
                                        gt_onsets=gt_onsets, activation= Superflux_activation_frames,start_exp=exp_start, end_exp=exp_end, hop_length= spf_hop , sr= spf_sr)    
    
 


    spf_pred_scnd_df = pd.DataFrame(spf_pred_scnd, columns=['onset_seconds']) 
    spf_pred_scnd_df.to_csv(os.path.join(Superflux_results_folder, chick +'_SPFpredictions.csv'), index=False)


####Evaluation for Double Threshold
    DBT_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'DBT_default_parameters_window' + str(evaluation_window))
    if not os.path.exists(DBT_results_folder):
        os.mkdir(DBT_results_folder)
    dbt_pred_scnd = onset_detectors.double_threshold(file)
    Fscore, precision, recall, TP, FP, FN = f_measure(gt_onsets, dbt_pred_scnd, window=evaluation_window)
    individual_fscore_list_DBT.append(Fscore)
    individual_precision_list_DBT.append(precision)
    individual_recall_list_DBT.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : chick , 'Algorithm':'Double Threshold',  'F-measure':Fscore, 'Precision':precision, 'Recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(DBT_results_folder , f'_{chick}_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(DBT_results_folder ,f'_{chick}_FN.csv'), index=False)
    with open(os.path.join(DBT_results_folder , f"_{chick}_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)


    dbt_pred_scnd_df = pd.DataFrame(dbt_pred_scnd, columns=['onset_seconds'])
    dbt_pred_scnd_df.to_csv(os.path.join(DBT_results_folder, chick +'_DBTpredictions.csv'), index=False)



    n_events_list.append(len(gt_onsets))


# compute weighted average
# print(
global_fmeasure_HFC_in_batch =  eval.compute_weighted_average(individual_fscore_list_HFC, n_events_list)
global_precision_HFC_in_batch = eval.compute_weighted_average(individual_precision_list_HFC, n_events_list)
global_recall_JFC_in_batch = eval.compute_weighted_average(individual_recall_list_HFC, n_events_list)

# compute weighted average
global_f1score_TPD_in_batch = eval.compute_weighted_average(individual_fscore_list_TPD, n_events_list)
global_precision_TPD_in_batch = eval.compute_weighted_average(individual_precision_list_TPD, n_events_list)
global_recall_TPD_in_batch = eval.compute_weighted_average(individual_recall_list_TPD, n_events_list)

# compute weighted average
global_f1score_NWPD_in_batch = eval.compute_weighted_average(individual_fscore_list_NWPD, n_events_list)
global_precision_NWPD_in_batch = eval.compute_weighted_average(individual_precision_list_NWPD, n_events_list)
global_recall_NWPD_in_batch = eval.compute_weighted_average(individual_recall_list_NWPD, n_events_list)


# compute weighted average
global_f1score_RCD_in_batch = eval.compute_weighted_average(individual_fscore_list_RCD, n_events_list)
global_precision_RCD_in_batch = eval.compute_weighted_average(individual_precision_list_RCD, n_events_list)
global_recall_RCD_in_batch = eval.compute_weighted_average(individual_recall_list_RCD, n_events_list)

# compute weighted average
global_f1score_Superflux_in_batch = eval.compute_weighted_average(individual_fscore_list_Superflux, n_events_list)
global_precision_Superflux_in_batch = eval.compute_weighted_average(individual_precision_list_Superflux, n_events_list)
global_recall_Superflux_in_batch = eval.compute_weighted_average(individual_recall_list_Superflux, n_events_list)

global_f1score_DBT_in_batch = eval.compute_weighted_average(individual_fscore_list_DBT, n_events_list)
global_precision_DBT_in_batch = eval.compute_weighted_average(individual_precision_list_DBT, n_events_list)
global_recall_DBT_in_batch = eval.compute_weighted_average(individual_recall_list_DBT, n_events_list)


globals_results_dict = { 'HFC': {'fmeaseure': global_fmeasure_HFC_in_batch, 'precision': global_precision_HFC_in_batch, 'recall': global_recall_JFC_in_batch},
                        'TPD': {'fmeaseure': global_f1score_TPD_in_batch, 'precision': global_precision_TPD_in_batch, 'recall': global_recall_TPD_in_batch},
                        'NWPD': {'fmeaseure': global_f1score_NWPD_in_batch, 'precision': global_precision_NWPD_in_batch, 'recall': global_recall_NWPD_in_batch},
                        'RCD': {'fmeaseure': global_f1score_RCD_in_batch, 'precision': global_precision_RCD_in_batch, 'recall': global_recall_RCD_in_batch},
                        'Superflux': {'fmeaseure': global_f1score_Superflux_in_batch, 'precision': global_precision_Superflux_in_batch, 'recall': global_recall_Superflux_in_batch},
                        'DBT': {'fmeaseure': global_f1score_DBT_in_batch, 'precision': global_precision_DBT_in_batch, 'recall': global_recall_DBT_in_batch}
                        }
with open(os.path.join(save_evaluation_results_path, "global_evaluation_results.json"), 'w') as fp:
    json.dump(globals_results_dict, fp)

print('done')



