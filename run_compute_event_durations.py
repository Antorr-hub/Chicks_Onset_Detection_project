import os
import glob
from tqdm import tqdm
import pandas as pd
import evaluation as my_eval
import numpy as np
import librosa as lb
import utils as ut



# create dictionary with the duration of the calls for each chick
chick_durations = {}
average_durations = {}
minimum_durations = {}
maximum_durations = {}
# Path to the folder containing the txt files to be evaluated
audio_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set'

# metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Training_set\\chicks_training_metadata.csv")

# Path to the folder where the evaluation results will be saved
save_evaluation_results_path = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Duration_from_testing_calls'
if not os.path.exists(save_evaluation_results_path):
    os.makedirs(save_evaluation_results_path)

offset_folder_predicted = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\normalised_data_only_inside_exp_window\\Testing_set'

n_events_list = []
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

predicted_offset_files = glob.glob(os.path.join(offset_folder_predicted, "*.txt"))

for file in tqdm(list_files):
    chick =os.path.basename(file)[:-4]  #'chick21_d0'
    # search in the offset folder the file with the same chick name
    pred_off_file = None
    for pred_offset_file in predicted_offset_files:
        if chick in pred_offset_file:
            pred_off_file = pred_offset_file
            break

    assert pred_off_file, "File not found"
    

    #offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))
    

    gt_onsets = my_eval.get_reference_onsets(pred_off_file)
    predicted_offsets = my_eval.get_reference_offsets(pred_off_file)
    # match the instances of the ground truth onsets with the instances of the estimated offsets in time 
  
    # # Retrieve experiment window boundaries
    # exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]
    # exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]
    # # Filter events to include only those within the experiment window


    # crete tuple of offsets, (gt, pred)
    # offsets_tuple = list(zip(gt_offsets, predicted_offsets))
    # Filter events to include only those within the experiment window
    # new_gt_onsets, new_offsets_tuple =  my_eval.discard_events_outside_experiment_window_offset_detection(exp_start, exp_end, gt_onsets, offsets_tuple)
    
    # new_gt_onsets =  gt_onsets[(gt_onsets >= exp_start) & (gt_onsets <= exp_end)]
    # new_pred_offsets = predicted_offsets[(predicted_offsets >= exp_start) & (predicted_offsets <= exp_end)]
    # new_gt_offsets = gt_offsets[(gt_offsets >= exp_start) & (gt_offsets <= exp_end)]


    events = list(zip(gt_onsets, predicted_offsets))
    # chick = os.path.basename(file)[:-4]
        
        
    durations = ut.calculate_durations(events)

    chick_durations[chick] = durations

    #save the durations in a csv  file for each chick
    df = pd.DataFrame(durations, columns = ['Duration'])
    df.to_csv(os.path.join(save_evaluation_results_path, chick + '_durations.csv'), index=False)

    #save the duration of alls the chicks in a csv with the chick name
    df = pd.DataFrame(durations, columns = ['Duration'])
    df['Chick'] = chick
    df = df[['Chick', 'Duration']]
    df.to_csv(os.path.join(save_evaluation_results_path, 'all_durations.csv'), mode='a', header=False, index=False)

    # compute the average duration of the calls for each chick

    for chick, durations in chick_durations.items():
        # compute the average duration of the calls for each chick
        average_duration = sum(durations) / len(durations)
        average_durations[chick] = average_duration
        # compute the minimum and maximum duration of the calls for each chick
        minimum_duration = min(durations)
        minimum_durations[chick] = minimum_duration

        maximum_duration = max(durations)
        maximum_durations[chick] = maximum_duration

    #save the average durations in a csv  file for all the chicks
    df = pd.DataFrame(average_durations.items(), columns = ['Chick', 'Average_duration'])
    df.to_csv(os.path.join(save_evaluation_results_path, 'average_durations.csv'), index=False)




# compute for all the chicks the average duration of the calls
global_average_duration = sum(average_durations.values()) / len(average_durations)

df = pd.DataFrame([global_average_duration], columns=['Global_average_duration'])
# Or use dictionary:
# Save the DataFrame to a CSV file
df.to_csv(os.path.join(save_evaluation_results_path, 'global_average_duration.csv'), index=False)

 

# compute the energy 