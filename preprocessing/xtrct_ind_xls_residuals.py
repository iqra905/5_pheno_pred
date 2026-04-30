import pandas as pd

#Read the text file
df = pd.read_csv('/vol/research/ucdatasets/gwas/data_files/37k.sample.txt', sep='\t')

# List of disease columns
disease_columns = ['t2dm_res', 'breacancer_res', 'crc_res', 'pros01_res', 'panca_res']


# keep rows with NA in any of the disease columns
df_cleaned = df.dropna(subset=disease_columns, how='all')
print(f"keepd {len(df) - len(df_cleaned)} rows with NA values in disease columns")
print(f"Remaining rows: {len(df_cleaned)}")

# Save the cleaned dataframe to a new text file
cleaned_file_name = '/vol/research/ucdatasets/gwas/data_files/37k.sample_cleaned.txt'
df_cleaned.to_csv(cleaned_file_name, sep='\t', index=False)

print(f"Cleaned file saved as {cleaned_file_name}")

# Define the output file mappings and columns to keep
output_files = {
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/t2d_res': {
        'chip_values': ['InterAct_Illumina660_cleaned.fam', 'InterAct_HumanCoreExome-24v1_cleaned.fam', 'InterAct_HumanCoreExome-12v1_cleaned.fam'],
        'keep_columns': ['ID1', 'ID2', 'missing', 't2dm_res', 'chip']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/brea_can_res': {
        'chip_values': ['brea01', 'brea02'],
        'keep_columns': ['ID1', 'ID2', 'missing', 'breacancer_res', 'chip']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/col_can_res': {
        'chip_values': ['crc'],
        'keep_columns': ['ID1', 'ID2', 'missing', 'crc_res', 'chip']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/pros_can_res': {
        'chip_values': ['pros01'],
        'keep_columns': ['ID1', 'ID2', 'missing', 'pros01_res', 'chip']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/pan_can_res': {
        'chip_values': ['panscan', 'panscan3'],
        'keep_columns': ['ID1', 'ID2', 'missing', 'panca_res', 'chip']
    }
}

# Process each output file
for output_file, file_info in output_files.items():
    # Filter the dataframe based on chip values
    filtered_df = df_cleaned[df_cleaned['chip'].isin(file_info['chip_values'])]
    
    # keep specified columns
    columns_to_keep = [col for col in filtered_df.columns if col in file_info['keep_columns']]
    filtered_df = filtered_df[columns_to_keep]

    # keep rows where the disease column is null
   # filtered_df = filtered_df.dropna(subset=[file_info['disease_column']])
    
    # Write the filtered dataframe to a new Excel file
    output_filename = f'{output_file}.xlsx'
    filtered_df.to_excel(output_filename, index=False)
    print(f"Generated: {output_filename}")

print("All files have been generated successfully.")
