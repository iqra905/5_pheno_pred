# # *************** Merging Results from a single folder *********************
import os
import pandas as pd
import argparse
from collections import OrderedDict

def merge_summary_csv(directory, output_path):
    # Initialize an ordered dictionary to store DataFrames
    dfs = OrderedDict()
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    experiment_folders = sorted(os.listdir(directory))
    
    if not experiment_folders:
        print(f"Warning: No folders found in directory '{directory}'.")
        return

    # Iterate over each folder in the directory
    for experiment_folder in experiment_folders:
        folder_path = os.path.join(directory, experiment_folder)
        
        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            # Path to the experiment_results.csv file in the experiment folder
            summary_file = os.path.join(folder_path, 'experiment_results.csv')

            # Check if the experiment_results.csv file exists
            if os.path.exists(summary_file):
                try:
                    # Read the CSV file into a DataFrame and add it to the dictionary
                    df = pd.read_csv(summary_file)
                    dfs[experiment_folder] = df
                    print(f"Loaded: {experiment_folder}")
                except Exception as e:
                    print(f"Error reading {summary_file}: {e}")
            else:
                print(f"Warning: experiment_results.csv not found in {experiment_folder}")

    if not dfs:
        print("No valid CSV files found to merge.")
        return

    # Concatenate all DataFrames in the dictionary in the order they were inserted
    merged_data = pd.concat(dfs.values(), ignore_index=True)
    # Sort the merged DataFrame based on the "test_auc" column in decreasing order
    # merged_data.sort_values(by="test_auc", ascending=False, inplace=True)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the merged data to a single CSV file
    try:
        merged_data.to_csv(output_path, index=False)
        print(f"Merged summary CSV file created successfully at: {output_path}")
        print(f"Total rows in merged file: {len(merged_data)}")
        print(f"Total experiments merged: {len(dfs)}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge experiment_results.csv files from multiple experiment folders into a single CSV file.") 
    parser.add_argument('-directory', type=str, help="Path to the directory containing experiment folders")
    parser.add_argument('-output_csv', type=str, help="Path where the merged CSV file will be saved")
    
    parser.add_argument('-sort-by', type=str, default=None, help="Column name to sort the merged data by (optional)")
    
    parser.add_argument('-ascending', action="store_true", help="Sort in ascending order (default is descending)")

    args = parser.parse_args()
    
    merge_summary_csv(args.directory, args.output_csv)
    
    if args.sort_by:
        try:
            df = pd.read_csv(args.output_csv)
            df.sort_values(by=args.sort_by, ascending=args.ascending, inplace=True)
            df.to_csv(args.output_csv, index=False)
            sort_order = "ascending" if args.ascending else "descending"
            print(f"Data sorted by '{args.sort_by}' in {sort_order} order.")
        except Exception as e:
            print(f"Error sorting data: {e}")

if __name__ == "__main__":
    main()

# # # *************** Merging Results from multiple folders with same parent folder *********************
# import os
# import pandas as pd
# from collections import OrderedDict

# def merge_all_thresholds(base_directory):
#     # List of threshold values
#     thresholds = ["0.01", "0.05", "0.1"]
    
#     # Initialize a list to store all DataFrames
#     all_dfs = []
    
#     # Iterate over each threshold
#     for threshold in thresholds:
#         # Construct the directory path for this threshold
#         directory = os.path.join(base_directory, threshold)
        
#         # Check if the directory exists
#         if not os.path.exists(directory):
#             print(f"Directory {directory} does not exist. fullping.")
#             continue
        
#         # Initialize an ordered dictionary to store DataFrames for this threshold
#         threshold_dfs = OrderedDict()
#         experiment_folders = sorted(os.listdir(directory))

#         # Iterate over each folder in the directory
#         for experiment_folder in experiment_folders:
#             # Check if the item in the directory is a folder
#             folder_path = os.path.join(directory, experiment_folder)
#             if os.path.isdir(folder_path):
#                 # Path to the experiment_results.csv file
#                 summary_file = os.path.join(folder_path, 'experiment_results.csv')
#                 #summary_file = os.path.join(folder_path, 'cv_results.csv')

#                 # Check if the file exists
#                 if os.path.exists(summary_file):
#                     # Read the file into a DataFrame
#                     df = pd.read_csv(summary_file)
#                     # Add a column for the threshold
#                     df['threshold'] = threshold
#                     threshold_dfs[experiment_folder] = df

#         # Concatenate all DataFrames for this threshold
#         if threshold_dfs:
#             threshold_data = pd.concat(threshold_dfs.values(), ignore_index=True)
#             all_dfs.append(threshold_data)
    
#     # Concatenate all threshold DataFrames
#     if all_dfs:
#         merged_data = pd.concat(all_dfs, ignore_index=True)
        
#         # Define output file path directly in the base directory
#         #output_file = os.path.join(base_directory, 'results_cnn_full_pruned_split_t2d_all_thresholds.csv')
#         #output_file = os.path.join(base_directory, 'results_mlp_full_summ_stats_pros_all_thresholds.csv')
#         output_file = os.path.join(base_directory, 'results_mlp_full_summ_stats_5D_all_thresholds_cov.csv')
        
#         # Write the merged data to a single CSV file
#         merged_data.to_csv(output_file, index=False)

#         print(f"Merged CSV file with all thresholds created successfully at {output_file}")
#     else:
#         print("No data found to merge.")

# # Base directory path (without the threshold)
# #base_directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_exp_5M_pruned_split/full/t2d'
# #base_directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_summ_stats/full'
# base_directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_summ_stats/full/covariates'


# # Call the function to merge all threshold results
# merge_all_thresholds(base_directory_path)