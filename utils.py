import numpy as np
import os
import pandas as pd
import librosa as lb

#calculate the duration of the calls from the ground truth onsets and offsets
def calculate_durations(events):   
    durations = []
    for event in events:
        duration = event[1] - event[0]
        durations.append(duration)
    return durations
