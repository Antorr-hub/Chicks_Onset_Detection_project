'''
The goal of an onset detection algorithm is to automatically determine when
notes are played in a piece of music.  The primary method used to evaluate
onset detectors is to first determine which estimated onsets are "correct",
where correctness is defined as being within a small window of a reference
onset.

Based in part on this script:

    https://github.com/CPJKU/onset_detection/blob/master/onset_evaluation.py

Conventions
-----------

Onsets should be provided in the form of a 1-dimensional array of onset
times in seconds in increasing order.

Metrics
-------

* :func:`mir_eval.onset.f_measure`: Precision, Recall, and F-measure scores
  based on the number of esimated onsets which are sufficiently close to
  reference onsets.
'''
from . import util_mod
import collections
import numpy as np
import warnings
import pandas as pd

# The maximum allowable beat time
MAX_TIME = 30000.



def validate(reference_onsets, estimated_onsets):
    """Checks that the input annotations to a metric look like valid onset time
    arrays, and throws helpful errors if not.

    Parameters
    ----------
    reference_onsets : np.ndarray
        reference onset locations, in seconds
    estimated_onsets : np.ndarray
        estimated onset locations, in seconds

    """
    # If reference or estimated onsets are empty, warn because metric will be 0
    if reference_onsets.size == 0:
        warnings.warn("Reference onsets are empty.")
    if estimated_onsets.size == 0:
        warnings.warn("Estimated onsets are empty.")
    for onsets in [reference_onsets, estimated_onsets]:
        util_mod.validate_events(onsets, MAX_TIME)



def f_measure(reference_onsets, estimated_onsets, window=.05):
    """Compute the F-measure of correct vs incorrectly predicted onsets.
    "Corectness" is determined over a small window.

    Examples
    --------
    >>> reference_onsets = mir_eval.io.load_events('reference.txt')
    >>> estimated_onsets = mir_eval.io.load_events('estimated.txt')
    >>> F, P, R = mir_eval.onset.f_measure(reference_onsets,
    ...                                    estimated_onsets)

    Parameters
    ----------
    reference_onsets : np.ndarray
        reference onset locations, in seconds
    estimated_onsets : np.ndarray
        estimated onset locations, in seconds
    window : float
        Window size, in seconds
        (Default value = .05)

    Returns
    -------
    f_measure : float
        2*precision*recall/(precision + recall)
    precision : float
        (# true positives)/(# true positives + # false positives)
    recall : float
        (# true positives)/(# true positives + # false negatives)

    """
    validate(reference_onsets, estimated_onsets, start_time=0, end_time=0)
    # If either list is empty, return 0s
    if reference_onsets.size == 0 or estimated_onsets.size == 0:
        return 0., 0., 0., [], estimated_onsets, reference_onsets
    # Compute the best-case matching between reference and estimated onset
    # locations
    matching = util_mod.match_events(reference_onsets, estimated_onsets, window)

    if len(matching) == 0:
        return 0., 0., 0., [], estimated_onsets, reference_onsets
    
    matched_reference = list(list(zip(*matching))[0])
    matched_estimated = list(list(zip(*matching))[1])

    ref_indexes = np.arange(len(reference_onsets))
    est_indexes = np.arange(len(estimated_onsets))
    unmatched_reference = set(ref_indexes) - set(matched_reference)
    unmatched_estimated = set(est_indexes) - set(matched_estimated)

    TP = estimated_onsets[matched_estimated]  
    FP = estimated_onsets[list(unmatched_estimated)]
    FN = reference_onsets[list(unmatched_reference)]

    precision = float(len(matching))/len(estimated_onsets)
    recall = float(len(matching))/len(reference_onsets)

    # Compute F-measure and return all statistics
    return util_mod.f_measure(precision, recall), precision, recall,TP, FP, FN
    


# def evaluate(reference_onsets, estimated_onsets, **kwargs):
#     """Compute all metrics for the given reference and estimated annotations.

#     Examples
#     --------
#     >>> reference_onsets = mir_eval.io.load_events('reference.txt')
#     >>> estimated_onsets = mir_eval.io.load_events('estimated.txt')
#     >>> scores = mir_eval.onset.evaluate(reference_onsets,
#     ...                                  estimated_onsets)

#     Parameters
#     ----------
#     reference_onsets : np.ndarray
#         reference onset locations, in seconds
#     estimated_onsets : np.ndarray
#         estimated onset locations, in seconds
#     kwargs
#         Additional keyword arguments which will be passed to the
#         appropriate metric or preprocessing functions.

#     Returns
#     -------
#     scores : dict
#         Dictionary of scores, where the key is the metric name (str) and
#         the value is the (float) score achieved.

#     """
#     # Compute all metrics
#     scores = collections.OrderedDict()

#     (scores['F-measure'],
#      scores['Precision'],
#      scores['Recall']) = util_mod.filter_kwargs(f_measure, reference_onsets,
#                                             estimated_onsets, **kwargs)
    
#     # (scores['F-measure'],
#     #  scores['Precision'],
#     #  scores['Recall']) = f_measure(reference_onsets, estimated_onsets, **kwargs)

#     return scores
