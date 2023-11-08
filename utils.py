import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import yaml


def visualize_onsets(gt_onsets, predicted_onsets, seconds, onset_function, file_name, plot_dir, algo_name): 
    plt.figure(figsize=(100, 5))

    plt.figure(figsize=(100,5))
    plt.plot(seconds, onset_function, alpha=0.8, label='Onset Function')
    #print(f"seconds: {seconds[:10]}")
    #print(f"onset_function: {onset_function[:10]}")
    #reference onsets
    for i in gt_onsets :
      plt.axvline(x=i, alpha=0.5, color="g")
    for i in predicted_onsets:
      plt.axvline(x=i, alpha=0.5, color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Onsets Visualization")
    
    # Create the "Onsets_plots" directory if it doesn't exist

    # Construct the output file path with the specified file name
    plot_filename = os.path.join(plot_dir, f"{file_name.split('.wav')[0]}_onset_plot_{algo_name}.png")
    print(f"plot_filename: {plot_filename}")
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved as {plot_filename}")


#################################################################################################################################


#latex_file_path = r"C:\Users\anton\OneDrive\Documenti\ODT_chicks\Training_set"
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
            
            #Precision         
            # precision_dict= {algo: result["Precision"] for algo, result in dict.items()}
            # data.append(precision_dict)

            # Recall
            # recall_dict= {algo: result["Recall"] for algo, result in dict.items()}
            # data.append(recall_dict)

            # df_pr= pd.DataFrame(precision_dict)
                     

            # Extract the data and store it in the dictionary without the specified prefix
            # entry_name = file_path.replace(prefix_to_remove, '')
            # data[entry_name] = json_data                


    # Create a DataFrame from the organized data
    df = pd.DataFrame.from_dict(data)
    df["Chick"] = [filename.replace(".json", "") for filename in os.listdir(folder_path) if filename.endswith(".json")]
    df= df[['Chick', 'High_frequency_content', 'Threshold_phase_deviation', 'Norm_weight_phase_deviation', 'Rectified_complex_domain', 'Superflux','Double_threshold_onset']]
  

    # Convert the DataFrame to a csv table
    csv_table  = df.to_csv()
    print(csv_table_table)

    # # # Save the LaTeX table to a file
    # with open(latex_file_path, 'w') as file:
    #     file.write(latex_table)

    # print(f"LaTeX table has been saved to '{latex_file_path}'.")

    return df

#################################################################################################################################

#################################################################################################################################
    
def load_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict