import pandas as pd
import numpy as np
import json

data_folder=r"C:\Users\anton\Chicks_Onset_Detection_project\Testing_not_norm_dataset_default_params_results"

#import json file from the folder
with open(data_folder+'\\'+'global_evaluation_results.json') as json_file:
  data = json.load(json_file)

df = pd.DataFrame(data).transpose()

#set directory to save the csv file

df.to_csv('TESTING_NOT_norm_dataset_default_params_results.csv', index_label='Method')
print(df)

latex_filename = 'TESTING_NOT_norm_dataset_default_params_results.tex'

df_subset = df.iloc[:, :4]

# Crea la tabella in formato LaTeX
latex_table = df_subset.to_latex(index=True, escape=False)

# Salva la tabella in formato LaTeX in un file .tex
with open(latex_filename, 'w') as latex_file:
    latex_file.write(latex_table)
