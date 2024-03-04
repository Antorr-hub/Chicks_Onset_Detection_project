import os
import glob
import onset_detection_algorithms_best_params as onset_detectors
from tqdm import tqdm
import pandas as pd
import evaluation as my_eval
from mir_eval_modified.onset import f_measure
import json
from visualization import visualize_activation_and_gt
from save_results import save_results_in_csv, save_global_results_latex
import matplotlib.pyplot as plt




#  WHEN running with parameters different from deafult, change here and name the results folder with the new parameters
EVAL_WINDOW = 0.1

# Parameters for the onset detection functions
HFC_parameters = {'hop_length': 441, 'sr':44100, 'spec_num_bands':15, 'spec_fmin': 2500, 'spec_fmax': 5000, 'spec_fref': 2800,
                  'pp_threshold':  1.8, 'pp_pre_avg':25, 'pp_post_avg':1, 'pp_pre_max':3, 'pp_post_max':2,'global shift': 0.1, 'double_onset_correction': 0.1}


Superflux_parameters= {'hop_length': 1024 // 2, 'n_fft': 2048 * 2, 'window': 0.12, 'fmin': 2050, 'fmax': 8000,'n_mels': 15, 'lag':3, 'max_size': 60,
                    'pp_pre_avg': 10, 'pp_post_avg': 10, 'pp_pre_max': 1, 'pp_post_max': 1,'pp_threshold': 0.01, 'pp_wait': 10, 
                     'global shift': 0, 'double_onset_correction': 0}
                       

# ###############################
#audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/'
#audio_folder = 'C:\\Users\\anton\\Vpa_experiment_data_normalise'  
#Data_normalised\\Validation_set'
audio_folder = 'C:\\Users\\anton\\Test_VPA_normalised'
#metadata = pd.read_csv("C:\\Users\\anton\\Test_VPA_normalised\\metadata_exp_autism.csv")
#metadata = pd.read_csv("/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/metadata.csv")
metadata = pd.read_csv("C:\\Users\\anton\\Test_VPA_normalised\\metadata_vpa_testing.csv")
#save_evaluation_results_path = "C:\\Users\\anton\\Vpa_experiment_data_normalise\\Results_Vpa_experiment"

#save_evaluation_results_path = r'/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Results/Default_parameters_evalWindow_0.1'

save_evaluation_results_path = r'C:\\Users\\anton\\Test_VPA_normalised\\counting_calls_results_on_testing'
#######################

if not os.path.exists(save_evaluation_results_path):
    os.makedirs(save_evaluation_results_path)



n_events_list = []
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))



for file in tqdm(list_files):

    # create folder for eaach chick results
    chick_folder = os.path.join(save_evaluation_results_path, os.path.basename(file[:-4]))
    if not os.path.exists(chick_folder):
        os.mkdir(chick_folder)

    # # get ground truth onsets
    gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    # # # discard events outside experiment window
    exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]   
    exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    
    chick = os.path.basename(file)[:-4]
    



    # #  High Frequency Content   
 
    HFC_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'HFC_default_parameters_window' + str(EVAL_WINDOW))
    if not os.path.exists(HFC_results_folder):
        os.mkdir(HFC_results_folder)
    
    # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
    hfc_pred_scnd,  HFC_activation_frames = onset_detectors.high_frequency_content(file, visualise_activation=True)

    #get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, hfc_pred_scnd, HFC_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end, 
                                                    gt_onsets, hfc_pred_scnd, HFC_activation_frames , hop_length=441, sr=44100)

    
    hfc_pred_scnd= my_eval.global_shift_correction(hfc_pred_scnd, 0.1)
    
    hfc_pred_scnd= my_eval.double_onset_correction(hfc_pred_scnd, correction= 0.1)
    


    visualize_activation_and_gt(plot_dir=HFC_results_folder,file_name=os.path.basename(file),onset_detection_funtion_name='High frequency content',
                             gt_onsets=gt_onsets, activation= HFC_activation_frames, start_exp=exp_start, end_exp=exp_end , hop_length=441, sr=44100)
    # save prediction to file
    hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])

    calls_detected_with_hfc = len(hfc_pred_scnd)

    hfc_predictions_seconds_df.to_csv(os.path.join(HFC_results_folder, chick +'_HFCpredictions.csv'), index=False)



###############Evaluation for Superflux
    Superflux_results_folder = os.path.join(save_evaluation_results_path, chick_folder, 'Superflux_default_parameters_window' + str(EVAL_WINDOW))
    if not os.path.exists(Superflux_results_folder):
        os.mkdir(Superflux_results_folder)

    spf_pred_scnd, Superflux_activation_frames, spf_hop, spf_sr = onset_detectors.superflux(file, visualise_activation=True)

    # get ground truth onsets, HFCpredictions_in_seconds, activation_frames inside experiment window
    gt_onsets, spf_pred_scnd, Superflux_activation_frames = my_eval.discard_events_outside_experiment_window(exp_start,exp_end,
                                                                                gt_onsets, spf_pred_scnd, Superflux_activation_frames, hop_length= spf_hop , sr= spf_sr)
      
    #spf_pred_scnd= my_eval.global_shift_correction(spf_pred_scnd, -0.05)
    spf_pred_scnd= my_eval.double_onset_correction(spf_pred_scnd, correction= 0)

    visualize_activation_and_gt(plot_dir=Superflux_results_folder,file_name=os.path.basename(file), onset_detection_funtion_name='Superflux', 
                                        gt_onsets=gt_onsets, activation= Superflux_activation_frames,start_exp=exp_start, end_exp=exp_end, hop_length= spf_hop , sr= spf_sr)    
    
    spf_pred_scnd_df = pd.DataFrame(spf_pred_scnd, columns=['onset_seconds']) 
    # count the number of onsets detected and save the results in a csv file
    calls_detected_with_superflux = len(spf_pred_scnd)
    spf_pred_scnd_df.to_csv(os.path.join(Superflux_results_folder, chick +'_SPFpredictions.csv'), index=False)




# Create DataFrame with calls detected and corresponding metadata including algorithm used
calls_detected_df_superflux = pd.DataFrame({'Calls_detected': calls_detected_with_superflux , 'Algorithm': 'Superflux'})
calls_detected_df_hfc = pd.DataFrame({'Calls_detected': calls_detected_with_hfc, 'Algorithm': 'HFC'})

file_names = [os.path.basename(file)[:-4] for file in list_files]
group = [metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Group'].values[0] for file in list_files]
sex = [metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Sex'].values[0] for file in list_files]

calls_detected_df_superflux['Filename'] = file_names
calls_detected_df_superflux['Group'] = group
calls_detected_df_superflux['Sex'] = sex

calls_detected_df_hfc['Filename'] = file_names
calls_detected_df_hfc['Group'] = group
calls_detected_df_hfc['Sex'] = sex

# Combine DataFrames
calls_detected_df = pd.concat([calls_detected_df_superflux, calls_detected_df_hfc], ignore_index=True)

# Save global calls detected in CSV
calls_detected_df.to_csv(os.path.join(save_evaluation_results_path, 'Calls_detected.csv'), index=False)

# Save Superflux and HFC results separately
calls_detected_df[calls_detected_df['Algorithm'] == 'Superflux'].to_csv(os.path.join(save_evaluation_results_path, 'Calls_detected_Superflux.csv'), index=False)
calls_detected_df[calls_detected_df['Algorithm'] == 'HFC'].to_csv(os.path.join(save_evaluation_results_path, 'Calls_detected_HFC.csv'), index=False)

# Compute average calls detected per group and sex and save plots
average_calls_per_group = calls_detected_df.groupby(['Group', 'Algorithm'])['Calls_detected'].mean().unstack()
average_calls_per_group.plot(kind='bar', title='Average calls detected per group')
plt.savefig(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_group.png'))
plt.close()

average_calls_per_sex = calls_detected_df.groupby(['Sex', 'Algorithm'])['Calls_detected'].mean().unstack()
average_calls_per_sex.plot(kind='bar', title='Average calls detected per sex')
plt.savefig(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_sex.png'))
plt.close()

# Compute and save average calls detected overall
average_calls_detected = calls_detected_df.groupby('Algorithm')['Calls_detected'].mean()
average_calls_detected.to_csv(os.path.join(save_evaluation_results_path, 'Average_calls_detected.csv'))

# Assuming `save_results_in_csv` is defined elsewhere and returns individual performances
individual_performances_chicks = save_results_in_csv(save_evaluation_results_path)

table_csv, latex_table = save_global_results_latex(save_evaluation_results_path)

print('done')
