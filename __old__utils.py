import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import yaml


def visualize_onsets(gt_onsets, predicted_onsets, seconds, onset_function, file_name, plot_dir, algo_name): 
    plt.figure(figsize=(100, 5))

    plt.figure(figsize=(100,5))
    plt.plot(seconds, onset_function, alpha=0.8, label='Onset Function')
    print(f"seconds: {seconds[:10]}")
    print(f"onset_function: {onset_function[:10]}")
    #reference onsets
    for i in gt_onsets :
      plt.axvline(x=i, alpha=0.5, color="g")
    for i in predicted_onsets:
      plt.axvline(x=i, alpha=0.5, color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Onsets Visualization")

    # Construct the output file path with the specified file name
    plot_filename = os.path.join(plot_dir, f"{file_name.split('.wav')[0]}_onset_plot_{algo_name}_.png")
    print(f"plot_filename: {plot_filename}")
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved as {plot_filename}")


#################################################################################################################################

folder_path = r"C:\Users\anton\Data_experiment\Data_normalised\Training_set"
prefix_to_remove = "C:\\Users\\anton\\Data_experiment\\Data_normalised\\Training_set\\"

def organize_json_files_to_dataframe(folder_path):
    
    # Iterate over the JSON files in the folder
    data =[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)


            # Read the JSON file
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            dict= list(json_data.values())[0]

            # F1-measure 
            f1_score_dict= {algo: result["F-measure"] for algo, result in dict.items()}
            data.append(f1_score_dict)            
            
    # Create a DataFrame from the organized data
    df = pd.DataFrame.from_dict(data)
    df["Chick"] = [filename.replace(".json", "") for filename in os.listdir(folder_path) if filename.endswith(".json")]
    df= df[['Chick', 'High_frequency_content', 'Threshold_phase_deviation', 'Norm_weight_phase_deviation', 'Rectified_complex_domain', 'Superflux','Double_threshold_onset']]
  

    # Convert the DataFrame to a csv table
    csv_table  = df.to_csv()
    print(csv_table_table)

    return df

#################################################################################################################################
# Double onsets correction function  for correcting the false positive of contact calls
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
      





#################################################################################################################################

def global_shift(predicted_onsets, shift):
    # compute global shift
    for i in predicted_onsets:
        #subtract a global shift of 0.01 ms or more  to all the predicted onsets
        global_shift= i - shift
        print("Global shift:", global_shift)
    return global_shift


#################################################################################################################################
    
def load_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict