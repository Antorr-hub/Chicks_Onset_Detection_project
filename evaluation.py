
import numpy as np
from mir_eval_modified import onset
import glob



def discard_events_outside_experiment_window(exp_start, exp_end, gt_events, predicted_events, predicted_events_frames, hop_length, sr): # TODO MODIFY THIS TO WORK WITH SUPERFLUX!!
 

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

def double_onset_correction(onsets_predicted, gt_onsets, correction= 0.020):
    '''Correct double onsets by removing onsets which are less than a given threshold in time.
    Args:
        onsets_predicted (list): List of predicted onsets.
        gt_onsets (list): List of ground truth onsets.
        correction (float): Threshold in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''    
    # Calculate interonsets difference
    gt_onsets = np.array(gt_onsets, dtype=float)

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
    return filtered_onsets
      
#######################** ALGORITHMS**#############################################
################################################################################### COMPARE WITH ABOVE??
#
# #  Compute filtering of onset detection function
def double_onsets_correction(onsets_predicted, gt_onsets, correction= 0.020):    
    # Calculate interonsets difference
    gt_onsets = np.array(gt_onsets, dtype=float)

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
    return filtered_onsets
      
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

    return corrected_predicted_onsets



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





def compute_precision_recall_curve(onset_detector_function, data_folder, list_peak_picking_thresholds, eval_window=0.1, **kwargs):

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


            predictions_scnd = onset_detector_function(file, pp_threshold =th) # TODO modify how we call this function to allow passing parameters contained in the **kwargs
            
            
            
            _, prec, rec, _,_,_ = onset.f_measure(gt_onsets, predictions_scnd, window=eval_window)
            individual_precision.append(prec)
            indicidual_recall.append(rec)


        av_precision = compute_weighted_average(individual_precision, n_events_list)
        av_recall = compute_weighted_average(indicidual_recall, n_events_list)
        av_precision_list.append(av_precision)
        av_recall_list.append(av_recall)
    

    return av_precision_list, av_recall_list


