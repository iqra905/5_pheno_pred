import pandas as pd

def compare_bim_files(file1, file2, output_file):
    # Read the .bim files
    df1 = pd.read_csv(file1, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    df2 = pd.read_csv(file2, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])

    # Add the original index as a column in df1
    df1['SNP_Index'] = df1.index

    # Find common rows
    common_rows = pd.merge(df1, df2, how='inner', on=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])

    # Select columns to save (including ORIGINAL_INDEX)
    columns_to_save = ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2', 'SNP_Index']
    common_rows = common_rows[columns_to_save]

    # Save to CSV
    common_rows.to_csv(output_file, index=False)

    print(f"Common rows saved to {output_file}")
    print(f"Total common rows: {len(common_rows)}")

# File paths
file1 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/t2d/t2d_merged.bim'
file2 = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/t2d/t2d_merged_pruned.bim'
output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/t2d/pruned_in_snps_t2d.csv'

# Run the comparison
compare_bim_files(file1, file2, output_file)