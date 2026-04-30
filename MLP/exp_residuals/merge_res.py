import os
import pandas as pd
from collections import OrderedDict

def merge_summary_csv(directory):
    # Initialize an ordered dictionary to store DataFrames
    dfs = OrderedDict()
    experiment_folders = sorted(os.listdir(directory))

    # Iterate over each folder in the directory
    for experiment_folder in experiment_folders:
        # Check if the item in the directory is a folder
        if os.path.isdir(os.path.join(directory, experiment_folder)):
            # Path to the summary.csv file in the experiment folder
            summary_file = os.path.join(directory, experiment_folder, 'experiment_results.csv')

            # Check if the summary.csv file exists
            if os.path.exists(summary_file):
                # Read the summary.csv file into a DataFrame and add it to the dictionary
                df = pd.read_csv(summary_file)
                dfs[experiment_folder] = df

    # Concatenate all DataFrames in the dictionary in the order they were inserted
    merged_data = pd.concat(dfs.values(), ignore_index=True)
    # Sort the merged DataFrame based on the "AUC Test" column in decreasing order
    merged_data.sort_values(by="test_r2", ascending=False, inplace=True)

    # Write the merged data to a single CSV file

    merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise/results_mlp_exp_res_chr_wise.csv', index=False)



    print("Merged summary CSV file sorted and created successchr_wisey.")

directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise'



# Call the function to merge and sort summary.csv files
merge_summary_csv(directory_path)