import json





def read_best_parameters_from_json(file_json):
    """Read the best parameters from a json file.

    Args:
        file_json (str): Path to the json file.  
        {"0": {"f_measure": 0.303183459951724, "num_bands": 15, "fmin": 1800, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "1": {"f_measure": 0.3228224594069739, "num_bands": 15, "fmin": 1800, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "2": {"f_measure": 0.3132764555574866, "num_bands": 15, "fmin": 1800, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "3": {"f_measure": 0.32948404763545464, "num_bands": 15, "fmin": 1800, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "4": {"f_measure": 0.38180050838654456, "num_bands": 15, "fmin": 2000, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "5": {"f_measure": 0.3731580888572922, "num_bands": 15, "fmin": 2000, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "6": {"f_measure": 0.3846611962524325, "num_bands": 15, "fmin": 2000, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "7": {"f_measure": 0.37836031966763883, "num_bands": 15, "fmin": 2000, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "8": {"f_measure": 0.14206895776790926, "num_bands": 24, "fmin": 1800, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "9": {"f_measure": 0.14238877247079804, "num_bands": 24, "fmin": 1800, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "10": {"f_measure": 0.14421156106830946, "num_bands": 24, "fmin": 1800, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "11": {"f_measure": 0.14401376122928913, "num_bands": 24, "fmin": 1800, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "12": {"f_measure": 0.15959726987037587, "num_bands": 24, "fmin": 2000, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "13": {"f_measure": 0.15971074978186206, "num_bands": 24, "fmin": 2000, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "14": {"f_measure": 0.16609539458979325, "num_bands": 24, "fmin": 2000, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "15": {"f_measure": 0.16614473183521838, "num_bands": 24, "fmin": 2000, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "16": {"f_measure": 0.14216477196746868, "num_bands": 32, "fmin": 1800, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "17": {"f_measure": 0.14217220414969362, "num_bands": 32, "fmin": 1800, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "18": {"f_measure": 0.14379887135740718, "num_bands": 32, "fmin": 1800, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "19": {"f_measure": 0.14395712096673344, "num_bands": 32, "fmin": 1800, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "20": {"f_measure": 0.1428402706853895, "num_bands": 32, "fmin": 2000, "fmax": 5000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "21": {"f_measure": 0.14247165493213138, "num_bands": 32, "fmin": 2000, "fmax": 5000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "22": {"f_measure": 0.14400298054342217, "num_bands": 32, "fmin": 2000, "fmax": 6000, "fref": 2500, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}, "23": {"f_measure": 0.14427579967534776, "num_bands": 32, "fmin": 2000, "fmax": 6000, "fref": 2800, "threshold": 1.8, "pre_avg": 0, "post_avg": 0, "pre_max": 1, "post_max": 1, "window": 0.1, "correction_shift": 0}}

    Returns:
        dict: Dictionary of best parameters.

    """
    with open(file_json, "r") as f:
        results_dict = json.load(f)


    # open json dictionary, from fmeasure column, pick the row with the highest fmeasure
    # and read the parameters from that row
    fmeasure_list = []
    for key in results_dict:
        fmeasure_list.append(results_dict[key]['f_measure'])
    best_fmeasure = max(fmeasure_list)
    best_parameters = {}
    for i, key in enumerate(results_dict):
        if results_dict[key]['f_measure'] == best_fmeasure:
            best_parameters[i] = results_dict[key]
            break
        else:
            continue    
    return best_parameters


if __name__ == "__main__":


    HFC_best_input_parameters = read_best_parameters_from_json('/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/chick_vocalisations/Chicks_Onset_Detection_project/grid_search_parameters/results_grid_searches/HFC_search_input_features.json')
    print(HFC_best_input_parameters)

    print('stop')
