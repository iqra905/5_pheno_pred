import pandas as pd
import numpy as np

def merge_xlsx_files(file1_path, file2_path, output_path):
    """
    Merge two Excel files based on SNP_index column, update specific columns from file2,
    remove rows from file1 that don't exist in file2, and sort by P-value
    """
    # Read Excel files
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)
    
    # Verify columns match
    if set(df1.columns) != set(df2.columns):
        raise ValueError("Column headers do not match between files")
    
    # Convert SNP_index to string type in both dataframes
    df1['SNP_index'] = df1['SNP_index'].astype(str)
    df2['SNP_index'] = df2['SNP_index'].astype(str)
    
    # Columns to update
    update_columns = ['SA', 'DC', 'GN', 'MLP_0.1', 'MLP_CHR_0.1', 'CNN_0.1', 
                     'CNN_SKIP_0.1', 'CNN_CHR_0.1', 'Methods_Count']
    
    # Keep only rows from df1 that exist in df2
    df1_filtered = df1[df1['SNP_index'].isin(df2['SNP_index'])]
    
    # Create a copy of filtered df1
    df1_updated = df1_filtered.copy()
    
    # Update values for each column separately to handle different data types
    df1_updated.set_index('SNP_index', inplace=True)
    df2.set_index('SNP_index', inplace=True)
    
    # Update existing rows while preserving data types
    for col in update_columns:
        if col in df2.columns:
            df1_updated[col] = df1_updated.index.map(df2[col])
    
    df1_updated.reset_index(inplace=True)
    
    # Find rows in df2 that aren't in df1 based on SNP_index
    missing_indices = set(df2.index) - set(df1_updated['SNP_index'])
    missing_rows = df2.loc[list(missing_indices)].reset_index()
    
    # Concatenate df1 with missing rows
    merged_df = pd.concat([df1_updated, missing_rows], ignore_index=True)
    
    # Convert P-value to numeric and sort
    merged_df['P-value'] = pd.to_numeric(merged_df['P-value'], errors='coerce')
    merged_df = merged_df.sort_values('P-value', ascending=True)
    
    # Save merged dataframe as Excel
    merged_df.to_excel(output_path, index=False)
    
    print(f"Original file 1 rows: {len(df1)}")
    print(f"Rows kept from file 1: {len(df1_filtered)}")
    print(f"Rows removed from file 1: {len(df1) - len(df1_filtered)}")
    print(f"Rows added from file 2: {len(missing_rows)}")
    print(f"Final merged rows: {len(merged_df)}")

# Example usage
if __name__ == "__main__":
    file1 = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/sig_prev/sig_snps_t2d_methods_common_1.xlsx"
    file2 = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/sig_snps_t2d_methods_common.xlsx"
    out_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/merged_output_t2d.xlsx"
    merge_xlsx_files(file1, file2, out_file)