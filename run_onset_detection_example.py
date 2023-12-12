
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import onset_detection_algorithms as onset_detectors
import mir_eval_modified as mir_eval_new
import mir_eval
import evaluation as eval
from visualization import visualize_activation_and_gt




audiofile = "C:\\Users\\anton\Data_experiment\\Data\\Training_set\\chick41_d0.wav"
#audiofile = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav"

save_predictions_path = r'./example_results/'
if not os.path.exists(save_predictions_path):
    os.mkdir(save_predictions_path)
    

HFCpredictions_in_seconds, HFCpredictions_in_frames = onset_detectors.high_frequency_content(audiofile, visualise_activation=True)      #using the default parameters! because they are defined in the function we do not need to write them here!

# TPDpredictions_in_seconds = onset_detectors.thresholded_phase_deviation(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

# NWPDpredictions_in_seconds = onset_detectors.normalized_weighted_phase_deviation(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

# RCDpredictions_in_seconds = onset_detectors.rectified_complex_domain(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!

# Spf_predictions_in_seconds, Spf_predictions_in_frames = onset_detectors.superflux(audiofile, ) #using the default parameters! because they are defined in the function we do not need to write them here!

# Dbt_predictions_in_seconds = onset_detectors.double_threshold(audiofile) #using the default parameters! because they are defined in the function we do not need to write them here!
# save prediction to file
# predictions_seconds_df = pd.DataFrame(predictions_in_seconds, columns=['onset_seconds'])
# predictions_seconds_df.to_csv(os.path.join(save_predictions_path, os.path.basename(audiofile[:-4]) +'_HFCpredictions.csv'), index=False)
print(f"predictions_in_seconds: {HFCpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {TPDpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {NWPDpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {RCDpredictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {Spf_predictions_in_seconds[:10]}")

# print(f"predictions_in_seconds: {Dbt_predictions_in_seconds[:10]}")


# ##evaluate
# get ground truth onsets
gt_onsets = eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))


# compute individual scores Fmeasure, precision, recall 
scores_hfc = mir_eval.onset.evaluate(gt_onsets, HFCpredictions_in_seconds, window=0.05)

# scores_tpd = mir_eval.onset.evaluate(gt_onsets, TPDpredictions_in_seconds, window=0.05)

# scores_nwpd = mir_eval.onset.evaluate(gt_onsets, NWPDpredictions_in_seconds, window=0.05)

# scores_rcd = mir_eval.onset.evaluate(gt_onsets, RCDpredictions_in_seconds, window=0.05)

# scores_spf = mir_eval.onset.evaluate(gt_onsets, Spf_predictions_in_seconds, window=0.05)

# scores_dbt = mir_eval.onset.evaluate(gt_onsets, Dbt_predictions_in_seconds, window=0.05)




print(f"Scores: {scores_hfc}")

# print(f"Scores: {scores_tpd}")

# print(f"Scores: {scores_nwpd}")

# print(f"Scores: {scores_rcd}")

# print(f"Scores: {scores_spf}")

# print(f"Scores: {scores_dbt}")

# visualise

visualize_activation_and_gt(plot_dir=save_predictions_path,file_name=os.path.basename(audiofile), onset_detection_funtion_name='HFC', gt_onsets=gt_onsets, activation=HFCpredictions_in_frames, hop_length=441, sr=44100)



print('done')








