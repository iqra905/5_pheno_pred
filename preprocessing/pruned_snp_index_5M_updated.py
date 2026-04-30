import pandas as pd

def compare_bim_files(file1, file2, global_file, output_file):
    # Read the .gen file (5 columns, no CM column)
    df1 = pd.read_csv(file1, sep=r'\s+', header=None, names=['CHR', 'SNP', 'BP', 'A1', 'A2'])
    
    # Read the .bim file (still has 6 columns including CM)
    df2 = pd.read_csv(file2, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    
    # Read the global .gen file
    global_df = pd.read_csv(global_file, sep=r'\s+', header=None, names=['CHR', 'SNP', 'BP', 'A1', 'A2'])
    
    # Add index columns
    df1['SNP_Index'] = df1.index
    global_df['Global_SNP_Index'] = global_df.index
    
    print(f"File1 has {len(df1)} rows")
    print(f"File2 has {len(df2)} rows")
    print(f"Global file has {len(global_df)} rows")
    
    # Replace "." SNP values with concatenated values
    mask1 = df1['SNP'] == '.'
    if mask1.any():
        print(f"Replacing {mask1.sum()} '.' SNP values in file1")
        df1.loc[mask1, 'SNP'] = df1.loc[mask1].apply(
            lambda row: f"{row['CHR']}:{row['BP']}:{row['A1']}:{row['A2']}", axis=1
        )
    
    mask2 = df2['SNP'] == '.'
    if mask2.any():
        print(f"Replacing {mask2.sum()} '.' SNP values in file2")
        df2.loc[mask2, 'SNP'] = df2.loc[mask2].apply(
            lambda row: f"{row['CHR']}:{row['BP']}:{row['A1']}:{row['A2']}", axis=1
        )
    
    mask_global = global_df['SNP'] == '.'
    if mask_global.any():
        print(f"Replacing {mask_global.sum()} '.' SNP values in global file")
        global_df.loc[mask_global, 'SNP'] = global_df.loc[mask_global].apply(
            lambda row: f"{row['CHR']}:{row['BP']}:{row['A1']}:{row['A2']}", axis=1
        )
    
    # Method 1: Match on CHR, BP, and compare alleles
    print("Performing position-based matching between file1 and file2...")
    merged_on_position = pd.merge(
        df1, df2, 
        on=['CHR', 'BP'], 
        suffixes=('_1', '_2')
    )
    
    # Keep only rows where alleles match (in any order)
    allele_matched = merged_on_position[
        ((merged_on_position['A1_1'] == merged_on_position['A1_2']) & 
         (merged_on_position['A2_1'] == merged_on_position['A2_2'])) |
        ((merged_on_position['A1_1'] == merged_on_position['A2_2']) & 
         (merged_on_position['A2_1'] == merged_on_position['A1_2']))
    ]
    
    # Method 2: Match on CHR, SNP, BP
    print("Performing SNP ID-based matching between file1 and file2...")
    snp_matched = pd.merge(
        df1, df2,
        on=['CHR', 'SNP', 'BP'],
        suffixes=('_1', '_2')
    )
    
    # Select and rename columns from each method
    result1 = allele_matched[['CHR', 'SNP_1', 'BP', 'A1_1', 'A2_1', 'SNP_Index']].rename(
        columns={'SNP_1': 'SNP', 'A1_1': 'A1', 'A2_1': 'A2'}
    )
    
    if len(snp_matched) > 0:
        result2 = snp_matched[['CHR', 'SNP', 'BP', 'A1_1', 'A2_1', 'SNP_Index']].rename(
            columns={'A1_1': 'A1', 'A2_1': 'A2'}
        )
        # Combine results
        matched_results = pd.concat([result1, result2])
    else:
        print("No matches found via SNP ID matching")
        matched_results = result1
    
    # Remove duplicates
    matched_results = matched_results.drop_duplicates(subset=['CHR', 'BP', 'SNP_Index'])
    
    print(f"Found {len(matched_results)} matches between file1 and file2")
    
    # Now match with global file to get Global_SNP_Index
    print("Matching with global file to get Global_SNP_Index...")
    
    # Create copies with swapped alleles for matching
    global_df_swapped = global_df.copy()
    global_df_swapped[['A1', 'A2']] = global_df_swapped[['A2', 'A1']]
    
    # Match by direct comparison
    global_match_direct = pd.merge(
        matched_results, global_df,
        on=['CHR', 'SNP', 'BP', 'A1', 'A2'],
        how='left'
    )
    
    # Match with swapped alleles for those that didn't match directly
    missing_after_direct = global_match_direct['Global_SNP_Index'].isna()
    if missing_after_direct.any():
        print(f"{missing_after_direct.sum()} rows need allele-swapped matching with global file")
        
        # Extract rows that didn't get a Global_SNP_Index
        still_to_match = global_match_direct[missing_after_direct].drop(columns=['Global_SNP_Index'])
        
        # Match with swapped alleles
        global_match_swapped = pd.merge(
            still_to_match, global_df_swapped,
            on=['CHR', 'SNP', 'BP'],
            how='inner',
            suffixes=('', '_global')
        )
        
        # Filter for matching alleles in reverse order
        global_match_swapped = global_match_swapped[
            (global_match_swapped['A1'] == global_match_swapped['A2_global']) & 
            (global_match_swapped['A2'] == global_match_swapped['A1_global'])
        ]
        
        # Keep only rows with valid matches
        if len(global_match_swapped) > 0:
            global_match_swapped = global_match_swapped[['CHR', 'SNP', 'BP', 'A1', 'A2', 'SNP_Index', 'Global_SNP_Index']]
            
            # Replace missing Global_SNP_Index values with values from swapped matching
            matched_with_direct = global_match_direct[~missing_after_direct]
            
            # Combine direct and swapped matches
            final_result = pd.concat([matched_with_direct, global_match_swapped])
        else:
            final_result = global_match_direct
    else:
        final_result = global_match_direct
    
    # Count how many rows have Global_SNP_Index
    global_idx_count = final_result['Global_SNP_Index'].notna().sum()
    print(f"Successfully matched {global_idx_count} out of {len(final_result)} rows with global file")
    
    # Save to CSV
    final_result.to_csv(output_file, index=False)
    
    print(f"Common rows saved to {output_file}")
    print(f"Total rows in output: {len(final_result)}")

# File paths
# file1 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq_gen_split_col_0.1.gen'
# file2 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split/gen_data_5M_filtered_plink_files/col/col_0.1_LDpruned.bim'
# global_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
# output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split/gen_data_5M_filtered_plink_files/col/col_0.1_LDpruned.csv'

# file1 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered/t2d/first_5_columns_gen_split_20_t2d_0.1.gen'
# file2 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered_plink_files/t2d/t2d_0.1_LDpruned.bim'
# global_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
# output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered_plink_files/t2d/t2d_0.1_LDpruned.csv'

file1 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_stratified_kfold/gen_data_5M_filtered/brea/first_5_columns_gen_split_brea_5_0.05.gen'
file2 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_stratified_kfold/gen_data_5M_filtered_plink_files/brea/brea_fold_5_0.05_LDpruned.bim'
global_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_stratified_kfold/gen_data_5M_filtered_plink_files/brea/brea_fold_5_0.05_LDpruned.csv'


# Run the comparison
compare_bim_files(file1, file2, global_file, output_file)



# import pandas as pd

# def compare_bim_files(file1, file2, output_file):
#     # Read the .bim files
#     df1 = pd.read_csv(file1, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
#     df2 = pd.read_csv(file2, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])

#     # Add the original index as a column in df1
#     df1['SNP_Index'] = df1.index

#     # Find common rows
#     common_rows = pd.merge(df1, df2, how='inner', on=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])

#     # Select columns to save (including ORIGINAL_INDEX)
#     columns_to_save = ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2', 'SNP_Index']
#     common_rows = common_rows[columns_to_save]

#     # Save to CSV
#     common_rows.to_csv(output_file, index=False)

#     print(f"Common rows saved to {output_file}")
#     print(f"Total common rows: {len(common_rows)}")

# # File paths

# file1 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_filtered_plink_files/col/col_0.1.bim'
# file2 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_filtered_plink_files/col/col_0.1_LDpruned.bim'
# output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_filtered_plink_files/col/col_0.1_LDpruned.csv'

# # Run the comparison
# compare_bim_files(file1, file2, output_file)