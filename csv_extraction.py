import os
import json
import pandas as pd
import numpy as np

csv_file_path = 'testing.csv' 
folder_path = r"C:\Users\anton\Data_experiment\Data_normalised\Testing_set"
prefix_to_remove = "C:\\Users\\anton\\Data_experiment\\Data_normalised\\Training_set\\"

def organize_json_files_to_dataframe(folder_path):
    
    # Iterate over the JSON files in the folder
    data_f1score =[]
    data_precision = []
    data_recall = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)


            # Read the JSON file
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            dict= list(json_data.values())[0]

            # F1-measure 
            f1_score_dict= {algo: result["F-measure"] for algo, result in dict.items()}
            data_f1score.append(f1_score_dict)            
            # Precision                   
            precision_dict= {algo: result["Precision"] for algo, result in dict.items()}
            data_precision.append(precision_dict)
            # Recall 
            recall_dict= {algo: result["Recall"] for algo, result in dict.items()}
            data_recall.append(recall_dict)
           

    # Create a DataFrame from the organized data F1 measure metric
    df_f1 = pd.DataFrame.from_dict(data_f1score)
    df_f1["Chick"] = [filename.replace(".json", "") for filename in os.listdir(folder_path) if filename.endswith(".json")]
    df_f1 = df_f1[['Chick', 'High_frequency_content', 'Threshold_phase_deviation', 'Norm_weight_phase_deviation', 'Rectified_complex_domain', 'Superflux', 'Double_threshold_onset']]
    df_f1 = df_f1.rename(columns={'Chick': 'Chick', 'High_frequency_content': 'Hfc_F1score', 'Threshold_phase_deviation': 'Tpd_F1score', 'Norm_weight_phase_deviation': 'Nwpd_F1score', 'Rectified_complex_domain': 'Rcd_F1score', 'Superflux': 'Superflux_F1score', 'Double_threshold_onset': 'DTo_F1score'})

    # Create a DataFrame from the organized data precision metric
    df_pr = pd.DataFrame.from_dict(data_precision)
    df_pr["Chick"] = [filename.replace(".json", "") for filename in os.listdir(folder_path) if filename.endswith(".json")]
    df_pr = df_pr[['Chick', 'High_frequency_content', 'Threshold_phase_deviation', 'Norm_weight_phase_deviation', 'Rectified_complex_domain', 'Superflux', 'Double_threshold_onset']]
    df_pr = df_pr.rename(columns={'Chick':'Chick', 'High_frequency_content': 'Hfc_Prec', 'Threshold_phase_deviation': 'Tpd_Prec', 'Norm_weight_phase_deviation': 'Nwpd_Prec', 'Rectified_complex_domain': 'Rcd_Prec', 'Superflux': 'Superflux_Prec', 'Double_threshold_onset': 'DTo_Prec'})

    # Create a DataFrame from the organized data recall metric
    df_rc = pd.DataFrame.from_dict(data_recall)
    df_rc["Chick"] = [filename.replace(".json", "") for filename in os.listdir(folder_path) if filename.endswith(".json")]
    df_rc = df_rc[['Chick', 'High_frequency_content', 'Threshold_phase_deviation', 'Norm_weight_phase_deviation', 'Rectified_complex_domain', 'Superflux', 'Double_threshold_onset']]
    df_rc = df_rc.rename(columns={'Chick':'Chick','High_frequency_content': 'Hfc_Recall', 'Threshold_phase_deviation': 'Tpd_Recall', 'Norm_weight_phase_deviation': 'Nwpd_Recall', 'Rectified_complex_domain': 'Rcd_Recall', 'Superflux': 'Superflux_Recall', 'Double_threshold_onset': 'DTo_Recall'})
    
    # Merge dataframes of different metrics in a single dataframe
    F1_pr_df= df_f1.merge(df_pr, left_on= "Chick", right_on="Chick")
    df_results= F1_pr_df.merge(df_rc, left_on="Chick", right_on= "Chick")
    # Convert the DataFrame to a LaTeX table
    
    df_results.to_csv(csv_file_path, index=False)


    return df_results

resulting_dataframe = organize_json_files_to_dataframe(folder_path)


# Print the DataFrame
print(resulting_dataframe)
