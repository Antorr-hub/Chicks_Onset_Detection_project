
import numpy as np
from mir_eval_modified import onset
import glob
import os
import pandas as pd





def discard_events_outside_experiment_window(exp_start, exp_end, gt_events, predicted_events, predicted_events_frames, hop_length= 441, sr= 44100): # TODO MODIFY THIS TO WORK WITH SUPERFLUX!!
 

    # Filter onsets within the specified time window
    new_gt_events =  gt_events[(gt_events >= exp_start) & (gt_events <= exp_end)]
    new_predicted_events = predicted_events[(predicted_events >= exp_start) & (predicted_events <= exp_end)]

    start_exp_frames = int(exp_start * sr / hop_length)
    end_exp_frames = int(exp_end * sr / hop_length)

    predicted_events_frames[ :start_exp_frames] =0
    predicted_events_frames[end_exp_frames:] =0
    predicted_events_frames = predicted_events_frames[:end_exp_frames+50]
    new_predicted_events_frames = predicted_events_frames 

      
    return  new_gt_events, new_predicted_events, new_predicted_events_frames


#######################** ALGORITHMS**#############################################
################################################################################### 
#
# #  Compute filtering of onset detection function
def double_onset_correction(onsets_predicted, correction=0.02):    
    
    if not correction == 0 or len(onsets_predicted) > 1:
        # Calculate the difference between consecutive onsets
        differences = np.diff(onsets_predicted)

        # Create a list to add the filtered onset and add a first value

        filtered_onsets = [onsets_predicted[0]]  #Add the first onset

        # Subtract all the onsets which are less than fixed threshold in time
        for i, diff in enumerate(differences):
            if diff >= correction:
            # keep the onset if the difference is more than the given selected time
                filtered_onsets.append(onsets_predicted[i + 1])
                #print the number of onsets predicted after correction
        return np.asarray(filtered_onsets)
    else:
        return onsets_predicted
      
############################################################################################
############################################################################################

def global_shift_correction(predicted_onsets, shift):
    '''subtract shift second to all the predicted onsets.
    Args:
        predicted_onsets (list): List of predicted onsets.
        shift (float): Global shift in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''
    # compute global shift
    corrected_predicted_onsets = []
    for po in predicted_onsets:
        #subtract a global shift of 0.01 ms or more  to all the predicted onsets
        if po - shift > 0: # to avoid negative onsets
            corrected_predicted_onsets.append(po - shift)
        else:
            continue

    return np.array(corrected_predicted_onsets)



# function to extract reference onsets
def get_reference_onsets(file_txt):
    """Extract reference onsets from a txt file.

    Args:
        file_txt (str): Path to the txt file.

    Returns:
        list: List of reference onsets.

    """
    gt_onsets = []
    with open(file_txt, "r",  encoding='latin-1') as file:
            rows = file.readlines()

    for row in rows:
        columns = row.split()

        if columns:
            first_value = float(columns[0])
            gt_onsets.append(first_value) 
    assert gt_onsets, "File cannot be read!"
    return np.array(gt_onsets)






def compute_weighted_average(scores_list, n_events_list):
    """Compute the weighted average of a list of scores.

    Args:
        scores_list (list): List of scores. Should work with any type of metric.
        n_events_list (list): List of number of events.

    Returns:
        float: Weighted average.

    """

    total_events = sum(n_events_list)
    weights_list = [n_events / total_events for n_events in n_events_list]
    return np.average(scores_list, weights=weights_list)





def compute_precision_recall_curve(onset_detector_function, data_folder, list_peak_picking_thresholds, exp_start, exp_end, eval_window=0.1,  hop_length= 441, sr= 44100):

    audiofiles = glob.glob(data_folder + '/*.wav')

    # Compute precision and recall for each threshold
    individual_precision = []
    indicidual_recall = []
    n_events_list = []
    av_precision_list = []
    av_recall_list = []
    for i, th in enumerate(list_peak_picking_thresholds):

        for file in audiofiles:

            gt_onsets = get_reference_onsets(file.replace('.wav', '.txt'))
            n_events_list.append(len(gt_onsets))


            predictions_scnd, predicted_events_frames = onset_detector_function(file, visualise_activation= True, pp_threshold=th, hop_length= hop_length, sr= sr)
                        
            
            
            gt_onsets, predictions_scnd, predicted_events_frames = discard_events_outside_experiment_window(exp_start,exp_end, 
                                                gt_onsets, predictions_scnd, predicted_events_frames, hop_length= hop_length, sr= sr)


            
            _, prec, rec, _,_,_ = onset.f_measure(gt_onsets, predictions_scnd, window=eval_window)
            individual_precision.append(prec)
            indicidual_recall.append(rec)


        av_precision = compute_weighted_average(individual_precision, n_events_list)
        av_recall = compute_weighted_average(indicidual_recall, n_events_list)
        av_precision_list.append(av_precision)
        av_recall_list.append(av_recall)
    

    return av_precision_list, av_recall_list


