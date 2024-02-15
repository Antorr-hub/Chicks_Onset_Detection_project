import os
import pandas as pd
import json
import glob
from os.path import basename

# Assuming df_f1 is already defined
df_f1 = pd.DataFrame(columns=["Chick", "Algorithm", "F1-measure", "Precision", "Recall"])

# Folder path containing all chick folders
chick_folder_path = r"C:\Users\anton\Chicks_Onset_Detection_project\Testing_norm_dataset_default_params_results"

chick_folder = basename(chick_folder_path)

# Iterate over the JSON files in the folder
for filename in glob.glob(f'{chick_folder_path}/**/*.json', recursive=True):
    if filename.endswith("_evaluation_results.json"):
        # Read the JSON file
        with open(filename, 'r') as file:
            json_data = json.load(file)

        # Extract relevant information from the JSON structure
        chick_name = json_data.get("audiofilename", "N/A")
        algorithm = json_data.get("Algorithm", "N/A")
        f_measure = json_data.get("F-measure", "N/A")
        precision = json_data.get("Precision", "N/A")
        recall = json_data.get("Recall", "N/A")

        # Add a new row to the DataFrame
        df_f1.loc[len(df_f1)] = [chick_name, algorithm, f_measure, precision, recall]

print(df_f1)

csv_filename = os.path.join(chick_folder_path, f"Overall_chicks_results_{chick_folder}.csv")

# Save the DataFrame as a CSV file
df_f1.to_csv(csv_filename, index=False)

print("DataFrame salvato come CSV:", csv_filename)