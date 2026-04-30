import pandas as pd

# Read the Excel file
df = pd.read_excel('/vol/research/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1.xlsx')

# List of disease columns (excluding gender-specific ones)
disease_columns = ['t2dm', 'breacancer', 'crc', 'pros01', 'panca']

# Remove rows with NA in any of the disease columns 
df_cleaned = df.dropna(subset=disease_columns, how='all')

# Define the output file mappings
output_files = {
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno_updated/t2d': {
        'disease_column': 't2dm',
        'remove_columns': ['breacancer', 'breacancer_fem', 'crc', 'pros01', 'pros01_male', 'panca']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno_updated/brea_can': {
        'disease_column': 'breacancer',
        'remove_columns': ['t2dm', 'breacancer_fem', 'crc', 'pros01', 'pros01_male', 'panca']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno_updated/col_can': {
        'disease_column': 'crc',
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem', 'pros01', 'pros01_male', 'panca']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno_updated/pros_can': {
        'disease_column': 'pros01',
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem',  'pros01_male', 'crc', 'panca']
    },
    '/vol/research/ucdatasets/gwas/data_files/disease_pheno_updated/pan_can': {
        'disease_column': 'panca',
        'remove_columns': ['t2dm', 'breacancer', 'breacancer_fem', 'crc', 'pros01', 'pros01_male']
    }
}

# Process each output file
for output_file, file_info in output_files.items():
   disease_col = file_info['disease_column']
   
   # Get disease cases (where disease column = 1)
   disease_cases = df_cleaned[df_cleaned[disease_col] == 1].copy()
   
   # Get controls (where ALL disease columns = 0)
   control_mask = df_cleaned[disease_columns].eq(0).all(axis=1)
   controls = df_cleaned[control_mask].copy()
   
   # Print debug information
   print(f"\nProcessing {output_file}:")
   print(f"Number of cases: {len(disease_cases)}")
   print(f"Number of controls: {len(controls)}")
   
   # Combine cases and controls
   filtered_df = pd.concat([disease_cases, controls], ignore_index=True)
   
   # Remove specified columns
   columns_to_keep = [col for col in filtered_df.columns if col not in file_info['remove_columns']]
   filtered_df = filtered_df[columns_to_keep]
   
   # Write the filtered dataframe to a new Excel file
   output_filename = f'{output_file}.xlsx'
   filtered_df.to_excel(output_filename, index=False)
   print(f"Generated {output_filename} with {len(filtered_df)} total rows")

print("\nAll files have been generated successfully.")