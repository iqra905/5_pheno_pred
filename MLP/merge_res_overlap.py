import os
import pandas as pd
from collections import OrderedDict
import re

def merge_summary_csv(directory):
    # List to store DataFrames with their numeric prefix
    df_list = []
    experiment_folders = os.listdir(directory)
    
    # Iterate over each folder in the directory
    for experiment_folder in experiment_folders:
        # Check if the item in the directory is a folder
        if os.path.isdir(os.path.join(directory, experiment_folder)):
            # Path to the experiment_results.csv file
            summary_file = os.path.join(directory, experiment_folder, 'experiment_results.csv')
            
            # Check if the file exists
            if os.path.exists(summary_file):
                # Extract the numeric prefix using regex to get just the number
                match = re.match(r'(\d+)_', experiment_folder)
                if match:
                    prefix = int(match.group(1))
                    
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(summary_file)
                    
                    # Add the experiment folder name as a column
                    df['Exp_ID'] = experiment_folder
                    
                    # Store the DataFrame and its numeric order
                    df_list.append((prefix, df))
    
    # Sort by the numeric prefix
    df_list.sort(key=lambda x: x[0])
    
    # Concatenate all DataFrames in the correct order
    merged_data = pd.concat([df for _, df in df_list], ignore_index=True)
    
    # Write the merged data to a CSV file
    output_path = os.path.join(directory, 'results_mlp_pruned_t2d_0.1_overlap_full_80_20_maf_0.15.csv')
    merged_data.to_csv(output_path, index=False)
    
    print(f"Merged CSV file created and sorted numerically from 1 to 13 successfully.")

directory_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis_t2d_full_80_20_maf_0.15/exp_mlp'
merge_summary_csv(directory_path)