import os
import pandas as pd
import glob

def merge_and_sort_csv_files(output_file):
    # List to store all dataframes
    all_dfs = []
    folder_name = f"/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_0.1/chr_wise/0_best_01_3_32_0.5_100_0.001_tanh_warmup_exponential_0.1_adamw_0.5/top_snps_csv"

    # Iterate through chromosomes 1 to 22
    for i in range(1, 23):
       
        file_path = os.path.join(folder_name, f"top_snps_chromosome_{i}.csv")
        
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        else:
            print(f"Warning: File not found in {folder_name}")

    # Concatenate all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Sort the combined dataframe by the 'weight' column in descending order
        combined_df = combined_df.sort_values(by='Importance', ascending=False)

        # Write the sorted dataframe to a new CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined and sorted data written to {output_file}")
    else:
        print("No data found to combine.")

# Specify the output file name
output_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_0.1/chr_wise/0_best_01_3_32_0.5_100_0.001_tanh_warmup_exponential_0.1_adamw_0.5/genome_wide_sig_snps.csv"

# Call the function
merge_and_sort_csv_files(output_file)