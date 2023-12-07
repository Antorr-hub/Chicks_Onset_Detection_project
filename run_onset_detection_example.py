
import pandas as pd
import os


import signal_processing_based_onset_detection_algorithms  as onset_detectors





audiofile = '/data/home/acw486/data_train_plus_val/chick41_d0.wav'
save_predictions_path = './example_results/'

predictions_seconds = onset_detectors.high_frequency_content_onset_detect(audiofile)

# save prediction to file
predictions_seconds_df = pd.DataFrame(predictions_seconds, columns=['onset_seconds'])
predictions_seconds_df.to_csv(os.path.join(save_predictions_path, audiofile +'_HFCpredictions.csv', index=False))



# visualise


# evaluate












