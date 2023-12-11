
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
# from visualization import visualize_onsets




# audiofile = "C:\\Users\\anton\Data_experiment\\Data\\Training_set\\chick41_d0.wav"
audiofile = "/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav"

save_predictions_path = r'./example_results/'
if not os.path.exists(save_predictions_path):
    os.mkdir(save_predictions_path)
    

predictions_in_seconds = onset_detectors.high_frequency_content(audiofile)      #using the default parameters! because they are defined in the function we do not need to write them here!

# save prediction to file
predictions_seconds_df = pd.DataFrame(predictions_in_seconds, columns=['onset_seconds'])
predictions_seconds_df.to_csv(os.path.join(save_predictions_path, os.path.basename(audiofile[:-4]) +'_HFCpredictions.csv'), index=False)


# ##evaluate
# get ground truth onsets
gt_onsets = eval.get_reference_onsets(audiofile.replace('.wav', '.txt'))


# compute individual scores Fmeasure, precision, recall 
scores = mir_eval.onset.evaluate(gt_onsets, predictions_in_seconds, window=0.05)

# TODO: Get TP, FP, FN:


print(f"Scores: {scores}")
# visualise



print('done')








