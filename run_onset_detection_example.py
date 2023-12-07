
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import onset_detection_algorithms as onset_detectors
import mir_eval
from evaluation import get_reference_onsets
from visualization import visualize_onsets




audiofile = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Data_train_val_normalised/chick41_d0.wav'
save_predictions_path = './example_results/'
if os.path.exists(save_predictions_path) == False:
    os.mkdir(save_predictions_path)
    
predictions_in_seconds = onset_detectors.high_frequency_content_onset_detect(audiofile) #using the default parameters!

# save prediction to file
predictions_seconds_df = pd.DataFrame(predictions_in_seconds, columns=['onset_seconds'])
predictions_seconds_df.to_csv(os.path.join(save_predictions_path, audiofile +'_HFCpredictions.csv', index=False))


# ##evaluate
# get ground truth onsets
gt_onsets = get_reference_onsets(audiofile.replace('.wav', '.txt'))

scores = mir_eval.onset.evaluate(gt_onsets, predictions_seconds_df, window=0.05)


# visualise










