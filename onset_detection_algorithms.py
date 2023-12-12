import os
import numpy as np
import librosa
import madmom
import glob



  
def superflux(file_name, mel_spec_hop_length=512, mel_spec_n_fft=4096, spec_num_bands=12, spec_fmin=1800, spec_fmax=6500, 
                           spec_fref=2500, pp_threshold= 2.5, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=1, pp_post_max=1 ): #hoplength = 1024//2 , n_fft=2048*2, window=0.12, fmin=2050, fmax=6000, n_mels=152
    # create variable to save onsets
    onset_sf = None
    #Load my file
    y, sr = librosa.load(file_name) # what are the default parameters for librosa.load?
    # Create the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=, hop_length = mel_spec_hop_length, window=0.12, fmin=2050, fmax=6000, n_mels=15)
    # detect onsets through spectral flux
    odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=1024 // 2, lag=5, max_size=50)
    # detect onsets through superflux
    onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf, sr=sr, hop_length=1024 // 2, units='time')
    #set hop length for conversion in seconds
    hop_length=1024 // 2
    # Compute the frames for the onset function
    frames = np.arange(0, len(odf_sf))
    # Calculate the time values for each frame
    seconds = frames * hop_length / sr

    # import the ground truth onsets
    # gt_onsets = [onset for onset in gt_onsets if onset.replace('.', '', 1).isdigit()]
    # Convert in float the ground truth onsets
    gt_onsets = [float(onsets) for onsets in gt_onsets]

    # Convert onset times from seconds to frame indices
    onset_frames = np.asarray(onset_sf * sr / hop_length) # Superflux in librosa gives onsets in seconds
    # print the number of onsets detected
    print("Number of onsets detected with Superflux algorithm:", len(onset_sf))
    # correction of the interonsets interval
    #onset_sf = double_onsets_correction(onset_sf, gt_onsets, correction= 0.020)
    print("Number of onsets detected with Superflux algorithm after correction:", len(onset_sf))

    # Save onsets to a text file
    fname_sans_path = os.path.basename(file_name)
    output_file = os.path.join(out_dir, f"Superflux_onsets_{fname_sans_path.split('.wav')[0]}.txt")
    with open(output_file, "w") as file:
        file.write("\n".join(map(str, onset_sf))) 
    print(f"The file has been saved in the output folder{out_dir} as {output_file}")    
    return onset_sf, output_file, odf_sf, seconds 
  
############################################################################################
############################################################################################


def high_frequency_content(file_name, hop_length=441, sr=44100, spec_num_bands=12, spec_fmin=1800, spec_fmax=6500, 
                           spec_fref=2500, pp_threshold= 2.5, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=1, pp_post_max=1 ):
    '''Compute the onsets using the high frequency content algorithm with madmom.
    Args:
        file_name (str): Path to the audio file.
        hop_length (int): Hop length in samples.
        sr (int): Sample rate.
        spec_num_bands (int): Number of filter bands.
        spec_fmin (int): Minimum frequency.
        spec_fmax (int): Maximum frequency.
        spec_fref (int): Reference frequency.
        pp_threshold (float): Threshold for peak picking.
        pp_pre_avg (int): Number of frames to average before peak.
        pp_post_avg (int): Number of frames to average after peak.
        pp_pre_max (int): Number of frames to search for local maximum before peak.
        pp_post_max (int): Number of frames to search for local maximum after peak.
    Returns:
        list: Onsets in seconds.
    '''

    spec_mdm = madmom.audio.spectrogram.FilteredSpectrogram(file_name,num_bands=spec_num_bands, fmin=spec_fmin , fmax=spec_fmax, fref=spec_fref, norm_filters=True, unique_filters=True)
    # Compute onset based on High frequency content with madmom
    hfc_ons = madmom.features.onsets.high_frequency_content(spec_mdm)
    # Applying the peak picking function to count number of onsets
    peaks = madmom.features.onsets.peak_picking(hfc_ons,threshold=pp_threshold, smooth=None, pre_avg=pp_pre_avg, post_avg=pp_post_avg, pre_max=pp_pre_max, post_max=pp_post_max)

    hfc_onsets_seconds =[(peak * hop_length / sr ) for peak in peaks ]    

    return np.array(hfc_onsets_seconds)






#  TODO: Modyfy the remaining functions to follow the same structure as the HFC function #############################






############################################################################################
#######################°°°THRESHOLDED PHASE DEVIATION°°°####################################
############################################################################################

# Define a function to run Thresholded phase deviation for ODT  
def tpd_ons_detect(file_name, hop_length=441, sr=44100, spec_num_bands=64, spec_fmin=1800, spec_fmax=6000, 
                   spec_fref=2500, pp_threshold= 0.95, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=10, pp_post_max=10):
    '''Compute the onsets using the thresholded phase deviation algorithm with madmom.
    Args:
        file_name (str): Path to the audio file.
        hop_length (int): Hop length in samples.
        sr (int): Sample rate.
        spec_num_bands (int): Number of filter bands.
        spec_fmin (int): Minimum frequency.
        spec_fmax (int): Maximum frequency.
        spec_fref (int): Reference frequency.
        alpha (float): Threshold for peak picking.
        pp_threshold (float): Threshold for peak picking.
        pp_pre_avg (int): Number of frames to average before peak.
        pp_post_avg (int): Number of frames to average after peak.
        pp_pre_max (int): Number of frames to search for local maximum before peak.
        pp_post_max (int): Number of frames to search for local maximum after peak.
        Returns:
        list: Onsets in seconds.
    '''

    # Create the filtered spectrogram
    spec_mdm = madmom.audio.spectrogram.FilteredSpectrogram(file_name, spec_num_bands=spec_num_bands, fmin=spec_fmin , fmax=spec_fmax, fref=spec_fref, norm_filters=True, unique_filters= True, circular_shift= True)

    # Compute phase deviation using madmom
    phase_ons_fn = madmom.features.onsets.phase_deviation(spectrogram=spec_mdm)

    # Assign the alpha value for thresholding
    alpha = 0.95
    # Apply thresholding on the phase deviation function
    phase_ons_fn[phase_ons_fn < alpha] = 0
    # Apply thresholding and peak picking  on the phase deviation function
    peaks = madmom.features.onsets.peak_picking(phase_ons_fn, threshold=pp_threshold, smooth=None, pre_avg=pp_pre_avg, post_avg=pp_post_avg, pre_max=pp_pre_max, post_max=pp_post_max)
    # Convert in seconds my onsets
    tpd_onsets_seconds= [(peak * hop_length / sr ) for peak in peaks] 

    return np.array(tpd_onsets_seconds)



############################################################################################
############################################################################################


############################################################################################
#######################°°°NORMALISED WEIGHTED PHASE DEVIATION°°°############################
############################################################################################
# Define a function to run Normalised weighted phase deviation (NWPD) for ODT
def nwpd_ons_detect(file_name, hop_length=441, sr=44100, pp_threshold= 0.92, pp_pre_avg=0, pp_post_avg=0, pp_pre_max=30, pp_post_max=30):  

    '''Compute the onsets using the normalized weighted phase deviation algorithm with madmom.
    Args:
        file_name (str): Path to the audio file.
        hop_length (int): Hop length in samples.
        sr (int): Sample rate.
        pp_threshold (float): Threshold for peak picking.
        pp_pre_avg (int): Number of frames to average before peak.
        pp_post_avg (int): Number of frames to average after peak.
        pp_pre_max (int): Number of frames to search for local maximum before peak.
        pp_post_max (int): Number of frames to search for local maximum after peak.
    Returns:
        list: Onsets in seconds.
    ''' 

    # Create the spectrogram using madmom
    madmom_spec = madmom.audio.spectrogram.Spectrogram(file_name, circular_shift= True)

    # Compute normalized weighted phase deviation using madmom
    nwpd_ons_fn = madmom.features.onsets.normalized_weighted_phase_deviation(madmom_spec, epsilon=2.220446049250313e-16)

    # Applying the peak picking function to count number of onsets # threshold= 0.92, smooth=None, pre_avg=0, post_avg=0, pre_max=30, post_max=30
    peaks = madmom.features.onsets.peak_picking(nwpd_ons_fn, threshold=pp_threshold, smooth=None, pre_avg=pp_pre_avg, post_avg=pp_post_avg, pre_max=pp_pre_max, post_max=pp_post_max)
    
    # Convert in seconds my onsets
    nwpd_onsets_seconds= [(peak * hop_length / sr ) for peak in peaks] 
      
    return np.array(nwpd_onsets_seconds)

############################################################################################
############################################################################################



############################################################################################
#######################°°°RECTIFIED COMPLEX DOMAIN°°°#######################################
############################################################################################
# Define a function to run Rectified complex domain (RCD) for ODT
def rcd_ons_detect(file_name, hop_length=441, sr=44100, pp_threshold= 50, pp_pre_avg=25, pp_post_avg=25, pp_pre_max=10, pp_post_max=10): 

    # Create the spectrogram using madmom
    madmom_spec = madmom.audio.spectrogram.Spectrogram(file_name, circular_shift= True)

    # Compute rectified complex domain onsets using madmom
    rcd_ons_fn = madmom.features.onsets.rectified_complex_domain(madmom_spec, diff_frames=None)

    # Applying the peak picking function with the current parameter values 
    peaks = madmom.features.onsets.peak_picking(rcd_ons_fn, threshold= pp_threshold, smooth=None, pre_avg=pp_pre_avg, post_avg=pp_post_avg, pre_max=pp_pre_max, post_max=pp_post_max)
    # Convert in seconds my onsets
    rcd_onsets_seconds= [(peak * hop_length / sr ) for peak in peaks] 
  
    return np.array(rcd_onsets_seconds)

############################################################################################
############################################################################################




############################################################################################
#######################°°°SUPERFLUX°°°######################################################
############################################################################################
# Define a function to run the Superflux algorithm for ODT
def superflux_ons_detect(file_name, spec_hop_length=1024 // 2, spec_n_fft=2048 *2, spec_window=0.12, spec_fmin=2050, spec_fmax=6000,
                         spec_n_mels=15, spec_lag=5, spec_max_size=50):
    '''Compute the onsets using the superflux algorithm with librosa
    Args:
        file_name (str): Path to the audio file.
        spec_hop_length (int): Hop length in samples.
        spec_n_fft (int): Number of FFT bins.
        spec_window (float): Window length in seconds.
        spec_fmin (int): Minimum frequency.
        spec_fmax (int): Maximum frequency.
        spec_n_mels (int): Number of Mel bands.
        spec_lag (int): Lag value for computing difference.
        spec_max_size (int): Maximum size of the onset detection function.
        Returns:
        list: Onsets in seconds.
        '''
    #Load my file
    y, sr = librosa.load(file_name)
    # Create the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft= spec_n_fft, hop_length= spec_hop_length, window= spec_window, fmin= spec_fmin, fmax=spec_fmax, n_mels= spec_n_mels)
    # detect onsets through spectral flux
    odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=sr, hop_length= spec_hop_length, lag= spec_lag, max_size= spec_max_size)
    # detect onsets through superflux
    onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf, sr=sr, hop_length= spec_hop_length, units='time')

    return np.array(onset_sf)
  
############################################################################################
############################################################################################



############################################################################################
#######################°°°DOUBLE THRESHOLD FUNCTION°°°######################################
############################################################################################
# double threshold approach to identify chicks' calls onsets
def double_thr_ons_detect(file_name, sr= 44100, hop_length=441, spec_n_fft=2048, spec_window=0.12, spec_fmin=2050, spec_fmax=6000):
    '''Compute the onsets using the double threshold algorithm with librosa
    Args:
        file_name (str): Path to the audio file.
        sr (int): Sample rate.
        hop_length (int): Hop length in samples.
        spec_n_fft (int): Number of FFT bins.
        spec_window (float): Window length in seconds.
        spec_fmin (int): Minimum frequency.
        spec_fmax (int): Maximum frequency.
    Returns:    
        list: Onsets in seconds.
    '''
    # Compute the STFT of the audio signal
    y, sr= librosa.load(file_name, sr= sr)

    # Compute the spectrogram magnitude
    spectrogram = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=spec_n_fft, window=spec_window, center=True, pad_mode='constant'))
    # Frequency range of interest (2 kHz to 5 kHz)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=spec_n_fft)
    #set the low frequency bin
    low_freq_bin = np.argmax(freq_bins >= 2000)
    #set the high frequency bin
    high_freq_bin = np.argmax(freq_bins >= 5800)

    # set the low threshold
    low_threshold = 100
    #set the high threshold
    high_threshold = 200

    # Initialize arrays to store onset times
    onset_times = []
    # for each time frame
    for t in range(spectrogram.shape[1]-1):
        # for each frequency bin
        for f in range(low_freq_bin, high_freq_bin):
            # if the value is higher than the threshold frequency value
            if f < spectrogram.shape[0]:
                # if the value is higher than the threshold frequency value OR if the value is higher than the threshold frequency value & 
                # the value is higher than the previous and the next value
                if (spectrogram[f, t] > high_threshold) | (spectrogram[f, t] > low_threshold and
                                                            (spectrogram[f, t] > spectrogram[f , t -1]) and
                                                            (spectrogram[f, t] > spectrogram[f , t +1])):

                    # Convert time frame index to seconds
                    onset_time = librosa.frames_to_time(t, sr=sr, hop_length=hop_length)
                    # Append the time to the array of onsets
                    onset_times.append(onset_time)
        
    return np.array(onset_times)

############################################################################################
############################################################################################