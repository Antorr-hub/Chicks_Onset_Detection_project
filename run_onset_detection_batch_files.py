import os

import glob
import onset_detection_algorithms as onset_detectors
import tqdm.tqdm
import pandas as pd
import mir_eval



audio_folder = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/'
save_predictions_path = './Results_normalised_data_default_parameters/'
list_files = glob.glob(os.path.join(audio_folder, "*.csv"))

individual_fscore_list = []
individual_precision_list = []
individual_recall_list = []
n_events_list = []
for file in tqdm(range(len(list_files))):

    pred_scnd = onset_detectors.high_frequency_content(file)
    # save prediction to file
    predictions_seconds_df = pd.DataFrame(pred_scnd, columns=['onset_seconds'])
    predictions_seconds_df.to_csv(os.path.join(save_predictions_path, file[:,-4] +'_HFCpredictions.csv', index=False))
    # ##evaluate
    # get ground truth onsets
    gt_onsets = eval.get_reference_onsets(file.replace('.wav', '.txt'))
    individual_scores = mir_eval.onset.evaluate(gt_onsets, pred_scnd, window=0.05)
    individual_fscore_list.append(individual_scores[0])
    individual_precision_list.append(individual_scores[1])
    individual_recall_list.append(individual_scores[2])

    n_events_list.append(len(gt_onsets))

# compute weighted average
print(global_f1score_in_batch = eval.compute_weighted_average(individual_fscore_list, n_events_list))
print(global_precision_in_batch = eval.compute_weighted_average(individual_precision_list, n_events_list))
print(global_recall_in_batch = eval.compute_weighted_average(individual_recall_list, n_events_list))




