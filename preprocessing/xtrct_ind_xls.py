import pandas as pd

# Read the Excel file
df = pd.read_excel('/vol/research/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1.xlsx')

# List of disease columns
disease_columns = ['t2dm', 'breacancer', 'breacancer_fem', 'crc', 'pros01', 'pros01_male', 'panca']


# Remove rows with NA in any of the disease columns
df_cleaned = df.dropna(subset=disease_columns, how='all')
print(f"Removed {len(df) - len(df_cleaned)} rows with NA values in disease columns")
print(f"Remaining rows: {len(df_cleaned)}")

# Save the cleaned dataframe to a new Excel file
cleaned_file_name = '/vol/research/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx'
df_cleaned.to_excel(cleaned_file_name, index=False)

print(f"Cleaned file saved as {cleaned_file_name}")

# Define the output file mappings and columns to remove
output_files = {
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/t2d': {
        'chip_values': ['InterAct_Illumina660_cleaned.fam', 'InterAct_HumanCoreExome-24v1_cleaned.fam', 'InterAct_HumanCoreExome-12v1_cleaned.fam'],
        'remove_columns': ['breacancer', 'breacancer_fem', 'crc', 'pros01', 'pros01_male', 'panca']
        #'disease_column': 't2dm'
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/brea_can': {
        'chip_values': ['brea01', 'brea02'],
        'remove_columns': ['t2dm', 'crc', 'pros01', 'pros01_male', 'panca']
        #'disease_column': 'breacancer'
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/col_can': {
        'chip_values': ['crc'],
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem', 'pros01', 'pros01_male', 'panca']
        #'disease_column': 'crc'
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/pros_can': {
        'chip_values': ['pros01'],
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem', 'crc', 'panca']
        #'disease_column': 'pros1'
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno/pan_can': {
        'chip_values': ['panscan', 'panscan3'],
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem', 'crc', 'pros01', 'pros01_male']
        #'disease_column': 'panca'
    }
}

# Process each output file
for output_file, file_info in output_files.items():
    # Filter the dataframe based on chip values
    filtered_df = df_cleaned[df_cleaned['chip'].isin(file_info['chip_values'])]
    
    # Remove specified columns
    columns_to_keep = [col for col in filtered_df.columns if col not in file_info['remove_columns']]
    filtered_df = filtered_df[columns_to_keep]

    # Remove rows where the disease column is null
   # filtered_df = filtered_df.dropna(subset=[file_info['disease_column']])
    
    # Write the filtered dataframe to a new Excel file
    output_filename = f'{output_file}.xlsx'
    filtered_df.to_excel(output_filename, index=False)
    print(f"Generated: {output_filename}")

print("All files have been generated successfully.")

