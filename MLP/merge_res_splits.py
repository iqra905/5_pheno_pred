import os
import pandas as pd
import matplotlib.pyplot as plt

def find_experimental_results(base_path):
    """
    Find all Experimental_results.csv files in split_* directories
    """
    result_files = []
    # List all directories in the base path
    for item in os.listdir(base_path):
        # Check if the item starts with 'split_' and is a directory
        if item.startswith('split_') and os.path.isdir(os.path.join(base_path, item)):
            # Construct path to potential results file
            result_path = os.path.join(base_path, item, "Exp_01", "experiment_results.csv")
            if os.path.isfile(result_path):
                result_files.append(result_path)
    
    return sorted(result_files)  # Sort to ensure consistent ordering

def merge_results(file_paths, output_path):
    """
    Merge all experimental results files and add a split number column
    """
    dfs = []
    for file_path in file_paths:
        # Extract split number from path - more robust extraction
        try:
            dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            split_num = int(dir_name.replace('split_', ''))
        except ValueError as e:
            print(f"Warning: Could not extract split number from {file_path}")
            continue
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            df['split'] = split_num
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files were found to merge!")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged results
    merged_df.to_csv(output_path, index=False)
    return merged_df

def create_plots(df, output_dir):
    """
    Create a bar plot showing test_auc and test_acc for each split
    """
    plt.figure(figsize=(12, 6))
    
    # Get the number of splits
    splits = sorted(df['split'].unique())
    x = range(len(splits))
    
    # Calculate mean AUC and ACC for each split
    aucs = [df[df['split'] == split]['test_auc'].mean() for split in splits]
    accs = [df[df['split'] == split]['test_acc'].mean() for split in splits]
    
    # Width of each bar
    width = 0.35
    
    # Create bars
    plt.bar([i - width/2 for i in x], aucs, width, label='AUC', color='blue')
    plt.bar([i + width/2 for i in x], accs, width, label='ACC', color='green')
    
    # Customize the plot
    plt.xlabel('Split Number')
    plt.ylabel('Score')
    plt.title('Test AUC and ACC across Splits - pros - Random_reverse Splits')
    plt.xticks(x, [f'Split {split}' for split in splits])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(aucs):
        plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(accs):
        plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_results_plots_pros_random_reverse.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define paths
    base_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/mlp_res_splits_random_reverse"
    output_path = os.path.join(base_path, "results_splits_pros_random_reverse.csv")
    
    # Find all experimental results files
    file_paths = find_experimental_results(base_path)
    
    if not file_paths:
        print("No Experimental_results.csv files found!")
        return
    
    print(f"Found {len(file_paths)} experimental results files:")
    for path in file_paths:
        print(f"  - {path}")
    
    # Merge results
    print("\nMerging results...")
    try:
        merged_df = merge_results(file_paths, output_path)
        print(f"Merged results saved to: {output_path}")
        
        # Create plots
        print("\nCreating plots...")
        create_plots(merged_df, base_path)
        print(f"Plots saved to: {base_path}/split_results_plots_pros_random_reverse.png")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def find_experimental_results(base_path):
#     """
#     Find all Experimental_results.csv files in split_* directories
#     """
#     result_files = []
#     # List all directories in the base path
#     for item in os.listdir(base_path):
#         # Check if the item starts with 'split_' and is a directory
#         if item.startswith('split_') and os.path.isdir(os.path.join(base_path, item)):
#             # Construct path to potential results file
#             result_path = os.path.join(base_path, item, "Exp_01", "experiment_results.csv")
#             if os.path.isfile(result_path):
#                 result_files.append(result_path)
    
#     return sorted(result_files)  # Sort to ensure consistent ordering

# def merge_results(file_paths, output_path):
#     """
#     Merge all experimental results files and add a split number column
#     """
#     dfs = []
#     for file_path in file_paths:
#         # Extract split number from path - more robust extraction
#         try:
#             dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
#             split_num = int(dir_name.replace('split_', ''))
#         except ValueError as e:
#             print(f"Warning: Could not extract split number from {file_path}")
#             continue
        
#         try:
#             # Read CSV
#             df = pd.read_csv(file_path)
#             df['split'] = split_num
#             dfs.append(df)
#         except Exception as e:
#             print(f"Error reading file {file_path}: {str(e)}")
#             continue
    
#     if not dfs:
#         raise ValueError("No valid data files were found to merge!")
    
#     # Concatenate all dataframes
#     merged_df = pd.concat(dfs, ignore_index=True)
    
#     # Save merged results
#     merged_df.to_csv(output_path, index=False)
#     return merged_df

# def create_plots(df, output_dir):
#     """
#     Create plots for test_auc and test_acc across splits
#     """
#     # Set style
#     plt.style.use('seaborn')
    
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
#     # Plot test_auc
#     sns.boxplot(data=df, x='split', y='test_auc', ax=ax1)
#     ax1.set_title('Test AUC across Splits')
#     ax1.set_xlabel('Split Number')
#     ax1.set_ylabel('Test AUC')
    
#     # Plot test_acc
#     sns.boxplot(data=df, x='split', y='test_acc', ax=ax2)
#     ax2.set_title('Test Accuracy across Splits')
#     ax2.set_xlabel('Split Number')
#     ax2.set_ylabel('Test Accuracy')
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'performance_plots.png'))
#     plt.close()

# def main():
#     # Define paths
#     base_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/mlp_res_splits"
#     output_path = os.path.join(base_path, "results_splits.csv")
    
#     # Find all experimental results files
#     file_paths = find_experimental_results(base_path)
    
#     if not file_paths:
#         print("No Experimental_results.csv files found!")
#         return
    
#     print(f"Found {len(file_paths)} experimental results files:")
#     for path in file_paths:
#         print(f"  - {path}")
    
#     # Merge results
#     print("\nMerging results...")
#     try:
#         merged_df = merge_results(file_paths, output_path)
#         print(f"Merged results saved to: {output_path}")
        
#         # Create plots
#         print("\nCreating plots...")
#         create_plots(merged_df, base_path)
#         print(f"Plots saved to: {base_path}/performance_plots.png")
#     except Exception as e:
#         print(f"Error during processing: {str(e)}")

# if __name__ == "__main__":
#     main()