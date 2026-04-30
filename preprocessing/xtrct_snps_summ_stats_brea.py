import pandas as pd
import numpy as np

def extract_significant_snps():
    # File paths
    file1_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen"
    file2_path = "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/bcac_meta_rs.txt"
    output_path = "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/extracted/brea/brea_0.1.csv"

    # Print file paths information
    print("File information:")
    print(f"File 1 path: {file1_path}")
    print(f"File 2 path: {file2_path}")
    print(f"Output path: {output_path}")
    print("-" * 50)
    
    # Read file1 - no headers, assigning column names
    col_names = ['chromosome', 'SNP_ID', 'bp', 'ref_allele', 'alt_allele']
    file1_df = pd.read_csv(file1_path, sep=r'\s+', header=None, names=col_names)

    # Print file1 information
    print("File 1 Information:")
    print(f"Number of rows: {len(file1_df)}")
    print(f"Columns: {', '.join(file1_df.columns)}")
    print("-" * 50)
    
    # Add SNP_Index column (enumeration starting from 0)
    file1_df['SNP_Index'] = np.arange(len(file1_df))
    
    # Read file2 - has headers
    file2_df = pd.read_csv(file2_path, sep=r'\s+')

    original_file2_length = len(file2_df)
    
    # Print file2 information
    print("File 2 Information:")
    print(f"Original number of rows: {len(file2_df)}")
    print(f"Columns: {', '.join(file2_df.columns)}")
    print("-" * 50)
    
    # Handle potential NaN values in the Pvalue column
    file2_df['Pvalue'] = pd.to_numeric(file2_df['bcac_onco_icogs_gwas_P1df'], errors='coerce')

    # Count NaN values in Pvalue column
    nan_count = file2_df['Pvalue'].isna().sum()
    print(f"Number of NaN values in Pvalue column: {nan_count}")
    
    # Filter file2 for P-values < 0.1
    file2_df = file2_df[file2_df['Pvalue'] < 0.1].copy()

    # Print filtered file2 information
    print(f"After filtering for P-value < 0.1: {len(file2_df)} rows")
    print("-" * 50)
    
    # Convert column types to ensure proper matching
    file1_df['chromosome'] = file1_df['chromosome'].astype(str)
    file2_df['Chr'] = file2_df['chr'].astype(str)
    
    file1_df['bp'] = file1_df['bp'].astype(int)
    file2_df['position'] = file2_df['position_b37'].astype(int)
    
    # Standardize allele case (convert all to uppercase)
    file1_df['ref_allele_upper'] = file1_df['ref_allele'].str.upper()
    file1_df['alt_allele_upper'] = file1_df['alt_allele'].str.upper()
    file2_df['Allele1_upper'] = file2_df['a0'].str.upper()
    file2_df['Allele2_upper'] = file2_df['a1'].str.upper()
    
    # Create a merged dataframe with direct matching
    print("Attempting direct allele matching...")
    merged_df = pd.merge(
        file1_df, 
        file2_df,
        left_on=['chromosome', 'bp', 'ref_allele_upper', 'alt_allele_upper'],
        right_on=['Chr', 'position', 'Allele1_upper', 'Allele2_upper'],
        how='inner'
    )

    print(f"Direct allele matching results: {len(merged_df)} matches")
    
    # If there are no direct matches, try with alleles swapped
    if len(merged_df) == 0:
        print("No direct matches found. Trying with alleles swapped...")
        merged_df = pd.merge(
            file1_df, 
            file2_df,
            left_on=['chromosome', 'bp', 'ref_allele_upper', 'alt_allele_upper'],
            right_on=['Chr', 'position', 'Allele2_upper', 'Allele1_upper'],
            how='inner'
        )
        print(f"Swapped allele matching results: {len(merged_df)} matches")
    
    # Create the final dataframe with required columns
    if len(merged_df) > 0:
        final_df = pd.DataFrame()
        final_df['chr'] = merged_df['chromosome']
        final_df['position'] = merged_df['bp']
        final_df['Allele1'] = merged_df['ref_allele']
        final_df['Allele2'] = merged_df['alt_allele']
        final_df['SNP_ID'] = merged_df['SNP_ID']
        final_df['SNP_Index'] = merged_df['SNP_Index']
        final_df['Pvalue'] = merged_df['Pvalue']
        
        # Print statistics about the final dataframe before deduplication
        print("Final dataframe statistics before deduplication:")
        print(f"Number of rows: {len(final_df)}")
        print("-" * 50)
        
        # Remove duplicate SNPs - keeping the one with lowest p-value in case of duplicates
        print(f"Before deduplication: {len(final_df)} SNPs")
        
        # Sort by p-value (ascending) to keep the most significant p-value for each SNP
        final_df = final_df.sort_values('Pvalue')
        
        # Count potential duplicates
        duplicate_count = len(final_df) - len(final_df.drop_duplicates(subset=['chr', 'position', 'Allele1', 'Allele2']))
        print(f"Number of potential duplicates: {duplicate_count}")
        
        # Drop duplicates based on SNP identifiers, keeping the first occurrence (lowest p-value)
        final_df = final_df.drop_duplicates(subset=['chr', 'position', 'Allele1', 'Allele2'], keep='first')
        
        # Print information about the deduplicated dataframe
        print(f"After deduplication: {len(final_df)} unique SNPs")
        print("-" * 50)
        
        # Save to CSV file
        final_df.to_csv(output_path, index=False)
        print(f"Created file at {output_path} with {len(final_df)} unique SNPs")
    else:
        print("No matching SNPs found. Output file was not created.")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"- Processed {len(file1_df)} SNPs from file 1")
    print(f"- Original dataset in file 2: {original_file2_length} SNPs")
    print(f"- Found {len(file2_df)} SNPs with P-value < 0.1 in file 2")
    print(f"- Initially matched {len(merged_df)} SNPs between the two files")
    if 'final_df' in locals():
        print(f"- Final dataset contains {len(final_df)} unique significant SNPs")
    print(f"- Processing complete!")

if __name__ == "__main__":
    extract_significant_snps()