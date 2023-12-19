import os
import glob
import onset_detection_algorithms as onset_detectors
from tqdm import tqdm
import pandas as pd
# import mir_eval
from mir_eval_modified import onset
import evaluation as eval
import json
from visualization import visualize_activation_and_gt



#audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/'
audio_folder = 'C:\\Users\\anton\\Data_experiment\\Data\\Training_set\\'
#save_predictions_path = './Results_normalised_data_default_parameters/'

save_evaluation_results_path = r'C:\Users\anton\Chicks_Onset_Detection_project\Results_data_default_parameters\Training'
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
    
    
 ###############Evaluation for High Frequency Content   
    # for alg in list_algorithms:
    HFC_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'HFC_default_parameters_window0.5')
    if not os.path.exists(HFC_results_folder):
        os.mkdir(HFC_results_folder)

    hfc_pred_scnd,  HFCpredictions_in_frames = onset_detectors.high_frequency_content(file, visualise_activation=True)
    gt_onsets, hfc_pred_scnd, HFCpredictions_in_frames = discard_events_outside_experiment_window(exp_start, exp_end, gt_onsets, hfc_pred_scnd, HFCpredictions_in_frames, hop_length, sr)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, hfc_pred_scnd, window=evaluation_window)
    individual_fscore_list_HFC.append(Fscore)
    individual_precision_list_HFC.append(precision)
    individual_recall_list_HFC.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'HFC',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(HFC_results_folder, 'TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(HFC_results_folder,'_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(HFC_results_folder, '_FN.csv'), index=False)
    with open(os.path.join(HFC_results_folder, f"{os.path.basename(file)[:-4]}_HFC_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=HFC_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='HFC', gt_onsets=gt_onsets, activation= HFCpredictions_in_frames, hop_length=441, sr=44100)



###############Evaluation for Thresholded Phase Deviation

    TPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'TPD_default_parameters_window0.5')
    if not os.path.exists(TPD_results_folder):
        os.mkdir(TPD_results_folder)

    tpd_pred_scnd, TPDpredictions_in_frames = onset_detectors.thresholded_phase_deviation(file, visualise_activation=True)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, tpd_pred_scnd, window=evaluation_window)
    individual_fscore_list_TPD.append(Fscore)
    individual_precision_list_TPD.append(precision)
    individual_recall_list_TPD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'TPD',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(TPD_results_folder, '_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(TPD_results_folder, '_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(TPD_results_folder,'_FN.csv'), index=False)
    with open(os.path.join(TPD_results_folder,f"{os.path.basename(file)[:-4]}_TPD_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=TPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='TPD', gt_onsets=gt_onsets, activation= TPDpredictions_in_frames, hop_length=441, sr=44100)    



###############Evaluation for Normalized Weighted Phase Deviation

    NWPD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'NWPD_default_parameters_window0.5')
    if not os.path.exists(NWPD_results_folder):
        os.mkdir(NWPD_results_folder)

    nwpd_pred_scnd, NWPDpredictions_in_frames = onset_detectors.normalized_weighted_phase_deviation(file, visualise_activation=True)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, nwpd_pred_scnd, window=evaluation_window)
    individual_fscore_list_NWPD.append(Fscore)
    individual_precision_list_NWPD.append(precision)
    individual_recall_list_NWPD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'NWPD',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(NWPD_results_folder,'_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(NWPD_results_folder, '_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(NWPD_results_folder, '_FN.csv'), index=False)
    with open(os.path.join(NWPD_results_folder,f"{os.path.basename(file)[:-4]}_NWPD_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=NWPD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='NWPD', gt_onsets=gt_onsets, activation= NWPDpredictions_in_frames, hop_length=441, sr=44100)    








###############Evaluation for Rectified Complex Domain
    RCD_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'RCD_default_parameters_window0.5')
    if not os.path.exists(RCD_results_folder):
        os.mkdir(RCD_results_folder)

    rcd_pred_scnd, RCDpredictions_in_frames = onset_detectors.rectified_complex_domain(file, visualise_activation=True)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, rcd_pred_scnd, window=evaluation_window)
    individual_fscore_list_RCD.append(Fscore)
    individual_precision_list_RCD.append(precision)
    individual_recall_list_RCD.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'RCD',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(RCD_results_folder, '_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(RCD_results_folder,'_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(RCD_results_folder, '_FN.csv'), index=False)
    with open(os.path.join(RCD_results_folder, '_evaluation_results.json'), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=RCD_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='RCD', gt_onsets=gt_onsets, activation= RCDpredictions_in_frames, hop_length=441, sr=44100)








###############Evaluation for Superflux
    Superflux_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'Superflux_default_parameters_window0.5')
    if not os.path.exists(Superflux_results_folder):
        os.mkdir(Superflux_results_folder)


    spf_pred_scnd, Superfluxpredictions_in_frames, spf_sr = onset_detectors.superflux(file, visualise_activation=True)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, spf_pred_scnd, window=evaluation_window)
    individual_fscore_list_Superflux.append(Fscore)
    individual_precision_list_Superflux.append(precision)
    individual_recall_list_Superflux.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Superflux',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(Superflux_results_folder, '_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(Superflux_results_folder, '_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(Superflux_results_folder, '_FN.csv'), index=False)
    with open(os.path.join(Superflux_results_folder, f"{os.path.basename(file)[:-4]}_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)

    visualize_activation_and_gt(plot_dir=Superflux_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Superflux', gt_onsets=gt_onsets, activation= Superfluxpredictions_in_frames, hop_length= 1024 // 2, sr= spf_sr)    


####Evaluation for Double Threshold
    DBT_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'DBT_default_parameters_window0.5')
    if not os.path.exists(DBT_results_folder):
        os.mkdir(DBT_results_folder)
    dbt_pred_scnd = onset_detectors.double_threshold(file)
    Fscore, precision, recall, TP, FP, FN = onset.f_measure(gt_onsets, dbt_pred_scnd, window=evaluation_window)
    individual_fscore_list_DBT.append(Fscore)
    individual_precision_list_DBT.append(precision)
    individual_recall_list_DBT.append(recall)
    # save Lists of TP, FP, FN for visualisation in sonic visualiser
    evaluation_results = { 'audiofilename' : os.path.basename(file), 'Algorithm':'Double Threshold',  'f_score':Fscore, 'precision':precision, 'recall': recall}
    TP_pd = pd.DataFrame(TP, columns=['TP'])
    TP_pd.to_csv(os.path.join(DBT_results_folder , '_TP.csv'), index=False)
    FP_pd = pd.DataFrame(FP, columns=['FP'])
    FP_pd.to_csv(os.path.join(DBT_results_folder ,'_FP.csv'), index=False)
    FN_pd = pd.DataFrame(FN, columns=['FN'])
    FN_pd.to_csv(os.path.join(DBT_results_folder ,'_FN.csv'), index=False)
    with open(os.path.join(DBT_results_folder , f"{os.path.basename(file)[:-4]}_evaluation_results.json"), 'w') as fp:
        json.dump(evaluation_results, fp)
  
    # save prediction to file
    hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])
    hfc_predictions_seconds_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_HFCpredictions.csv'), index=False)

    tpd_predictions_seconds_df = pd.DataFrame(tpd_pred_scnd, columns=['onset_seconds'])
    tpd_predictions_seconds_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_TPDpredictions.csv'), index=False)

    nwpd_pred_scnd_df = pd.DataFrame(nwpd_pred_scnd, columns=['onset_seconds'])
    nwpd_pred_scnd_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_NWPDpredictions.csv'), index=False)

    rcd_pred_scnd_df = pd.DataFrame(rcd_pred_scnd, columns=['onset_seconds'])
    rcd_pred_scnd_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_RCDpredictions.csv'), index=False)

    spf_pred_scnd_df = pd.DataFrame(spf_pred_scnd, columns=['onset_seconds'])
    spf_pred_scnd_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_SPFpredictions.csv'), index=False)

    dbt_pred_scnd_df = pd.DataFrame(dbt_pred_scnd, columns=['onset_seconds'])
    dbt_pred_scnd_df.to_csv(os.path.join(save_evaluation_results_path, file[:-4] +'_DBTpredictions.csv'), index=False)



    n_events_list.append(len(gt_onsets))


# compute weighted average
print(global_f1score_HFC_in_batch = eval.compute_weighted_average(individual_fscore_list_HFC, n_events_list))
print(global_precision_HFC_in_batch = eval.compute_weighted_average(individual_precision_list_HFC, n_events_list))
print(global_recall_JFC_in_batch = eval.compute_weighted_average(individual_recall_list_HFC, n_events_list))

# compute weighted average
print(global_f1score_TPD_in_batch = eval.compute_weighted_average(individual_fscore_list_TPD, n_events_list))
print(global_precision_TPD_in_batch = eval.compute_weighted_average(individual_precision_list_TPD, n_events_list))
print(global_recall_TPD_in_batch = eval.compute_weighted_average(individual_recall_list_TPD, n_events_list))

# compute weighted average
print(global_f1score_NWPD_in_batch = eval.compute_weighted_average(individual_fscore_list_NWPD, n_events_list))
print(global_precision_NWPD_in_batch = eval.compute_weighted_average(individual_precision_list_NWPD, n_events_list))
print(global_recall_NWPD_in_batch = eval.compute_weighted_average(individual_recall_list_NWPD, n_events_list))


# compute weighted average
print(global_f1score_RCD_in_batch = eval.compute_weighted_average(individual_fscore_list_RCD, n_events_list))
print(global_precision_RCD_in_batch = eval.compute_weighted_average(individual_precision_list_RCD, n_events_list))
print(global_recall_RCD_in_batch = eval.compute_weighted_average(individual_recall_list_RCD, n_events_list))

# compute weighted average
print(global_f1score_Superflux_in_batch = eval.compute_weighted_average(individual_fscore_list_Superflux, n_events_list))
print(global_precision_Superflux_in_batch = eval.compute_weighted_average(individual_precision_list_Superflux, n_events_list))
print(global_recall_Superflux_in_batch = eval.compute_weighted_average(individual_recall_list_Superflux, n_events_list))

print(global_f1score_DBT_in_batch = eval.compute_weighted_average(individual_fscore_list_DBT, n_events_list))
print(global_precision_DBT_in_batch = eval.compute_weighted_average(individual_precision_list_DBT, n_events_list))
print(global_recall_DBT_in_batch = eval.compute_weighted_average(individual_recall_list_DBT, n_events_list))


print('done')



