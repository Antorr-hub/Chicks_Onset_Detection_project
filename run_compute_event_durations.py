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
audio_folder = 'C:\\Users\\anton\\Data_normalised\\Training_set'

metadata = pd.read_csv("C:\\Users\\anton\\Data_normalised\\Training_set\\chicks_training_metadata.csv")

# Path to the folder where the evaluation results will be saved
save_evaluation_results_path = r'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Duration_from_estimated_offsets_training_calls'
if not os.path.exists(save_evaluation_results_path):
    os.makedirs(save_evaluation_results_path)

offset_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Estimated_offsets\\training_offset'

n_events_list = []
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

offset_files = glob.glob(os.path.join(offset_folder, "*.txt"))

for file in tqdm(list_files):
    chick =os.path.basename(file)[:-4]  #'chick21_d0'
    # search in the offset folder the file with the same chick name
    off_file = None
    for offset_file in offset_files:
        if chick in offset_file:
            off_file = offset_file
            break
    
    assert off_file, "File not found"
    

    #offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))
    offsets = my_eval.get_external_reference_offsets(off_file)
    gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    #gt_offsets = my_eval.get_reference_offsets(
    # match the instances of the ground truth onsets with the instances of the estimated offsets in time 
  
    # Retrieve experiment window boundaries
    exp_start = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['Start_experiment_sec'].values[0]
    exp_end = metadata[metadata['Filename'] == os.path.basename(file)[:-4]]['End_experiment_sec'].values[0]

    # Filter events to include only those within the experiment window

    
    new_gt_onsets =  gt_onsets[(gt_onsets >= exp_start) & (gt_onsets <= exp_end)]
    new_offsets = offsets[(offsets >= exp_start) & (offsets <= exp_end)]

    events = list(zip(new_gt_onsets, new_offsets))
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