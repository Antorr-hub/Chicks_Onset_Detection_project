
import os
import matplotlib.pyplot as plt



def visualize_onsets(gt_onsets, predicted_onsets, seconds, onset_function, file_name, plot_dir, algo_name):   # TODO REVIEW FUNCTION! 
    plt.figure(figsize=(100, 5))

    plt.figure(figsize=(100,5))
    plt.plot(seconds, onset_function, alpha=0.8, label='Onset Function') # label = hardcoded string!!!!
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



# def generate_annotation_files_for_sonic_visualiser(audiofilename, lists_onsets_to_print, lists_labels_to_print, annotations_file_path ):

#     # lists_onsets_to_print = [gt_onsets, predicted_onsets_HFC, predicted_onsets_TPD]


#     # generate single file in the format:

#     # Time, label,
#     # 0.0, GT
#     # 0.02, predicted_HFC
#     # 0.01, predicted_TPD

#     # annotations_file = 


#     return annotations_file