import pandas as pd
import os
import numpy as np
from pathlib import Path

# Define paths
input_file = "/vol/research/ucdatasets/gwas/data_files/disease_pheno/t2d.xlsx"
output_dir = "/vol/research/ucdatasets/gwas/data_files/disease_pheno/country_coded/t2d"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the Excel file
print(f"Reading data from {input_file}...")
df = pd.read_excel(input_file)

# Check if 'epicid' column exists
if 'epicid' not in df.columns:
    raise ValueError("The 'epicid' column was not found in the Excel file.")

# Extract the first character from the epicid column
df['first_char'] = df['epicid'].astype(str).str[0]

# Get unique first characters
unique_first_chars = df['first_char'].unique()
print(f"Found {len(unique_first_chars)} unique first characters in epicid: {sorted(unique_first_chars)}")

# Create separate files for each first character
for char in unique_first_chars:
    if pd.isna(char) or char == '':
        print("Skipping empty or NaN values...")
        continue
        
    # Filter rows with current first character
    subset = df[df['first_char'] == char].copy()
    
    # Remove the temporary 'first_char' column before saving
    subset = subset.drop(columns=['first_char'])
    
    # Define output file path
    output_file = os.path.join(output_dir, f"t2d_{char}.xlsx")
    
    # Save to Excel file
    print(f"Saving {len(subset)} rows to {output_file}...")
    subset.to_excel(output_file, index=False)
    
    # Print sample info for verification
    print(f"  Sample epicids: {subset['epicid'].head(3).tolist()}")

print(f"Processing complete. Files saved to {output_dir}")