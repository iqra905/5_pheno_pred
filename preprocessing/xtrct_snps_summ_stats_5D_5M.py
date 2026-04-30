import pandas as pd
import numpy as np
import os
import gc
import time
import psutil
import warnings
from datetime import datetime

# Suppress specific pandas warnings about mixed data types
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

def memory_usage():
    """Return the memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return f"{mem:.2f} MB"

def process_disease_data(disease_name, file1_path, file2_path, output_path, column_mappings, 
                         use_threshold=False, pvalue_threshold=0.1, stats_file=None):
    """
    Process GWAS data for a specific disease, finding overlaps between files
    using chunked processing to handle large files and saving in CSV format.
    
    Args:
        disease_name: Name of the disease being processed
        file1_path: Path to reference SNP file
        file2_path: Path to disease-specific SNP file
        output_path: Path to save output CSV file
        column_mappings: Dictionary mapping column names
        use_threshold: Whether to apply p-value thresholding (default: False)
        pvalue_threshold: P-value threshold to use if use_threshold=True (default: 0.1)
        stats_file: Path to statistics file (optional)
    """
    start_time = time.time()
    
    # Statistics tracking
    stats = {
        "disease_name": disease_name,
        "file1_total_snps": 0,
        "file2_total_snps": 0,
        "overlapping_snps": 0,            # SNPs that match between files
        "file2_nonoverlapping_snps": 0,   # SNPs in disease file with no match in reference file
        "file1_nonoverlapping_snps": 0,   # SNPs in reference file with no match in disease file
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_threshold": use_threshold
    }
    
    # Add p-value threshold statistics if using thresholding
    if use_threshold:
        stats.update({
            "p_value_threshold": pvalue_threshold,
            "file2_significant_snps": 0,  # SNPs that pass p-value threshold
            "filtered_snps": 0,           # SNPs filtered due to p-value threshold
        })
    
    # Print file paths information
    print(f"\nProcessing {disease_name}:")
    print(f"File 1 path: {file1_path}")
    print(f"File 2 path: {file2_path}")
    print(f"Output path: {output_path}")
    if use_threshold:
        print(f"P-value thresholding enabled with threshold: {pvalue_threshold}")
    else:
        print("P-value thresholding disabled - using direct overlap")
    print("-" * 50)
    
    # Extract column names from the mappings
    chr_col = column_mappings.get('chromosome')
    pos_col = column_mappings.get('position')
    allele1_col = column_mappings.get('allele1')
    allele2_col = column_mappings.get('allele2')
    pval_col = column_mappings.get('pvalue')
    
    # Print the identified column mappings
    print(f"Column mappings for {disease_name}:")
    print(f"Chromosome: {chr_col}")
    print(f"Position: {pos_col}")
    print(f"Allele 1: {allele1_col}")
    print(f"Allele 2: {allele2_col}")
    print(f"P-value: {pval_col}")
    print("-" * 50)
    
    # Exit if we couldn't find the necessary columns
    if None in [chr_col, pos_col, allele1_col, allele2_col, pval_col]:
        print(f"Error: Missing required column mappings for {disease_name}.")
        return stats
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First, get the column names from file2 and count total rows
    try:
        # Count total rows in file2 (this might be slow for large files)
        print("Counting total SNPs in disease file")
        with open(file2_path, 'r') as f:
            for i, _ in enumerate(f):
                pass
        stats["file2_total_snps"] = i + 1  # +1 because i is 0-indexed
        print(f"Total SNPs in {disease_name} file: {stats['file2_total_snps']}")
        
        # Get headers from file2
        file2_sample = pd.read_csv(file2_path, sep=r'\s+', nrows=5, low_memory=False)
        
        # Verify that the required columns exist
        for col_name, col in [('Chromosome', chr_col), ('Position', pos_col), 
                          ('Allele 1', allele1_col), ('Allele 2', allele2_col),
                          ('P-value', pval_col)]:
            if col not in file2_sample.columns:
                print(f"Error: Column '{col}' specified as {col_name} does not exist in the {disease_name} file.")
                print(f"Available columns: {', '.join(file2_sample.columns)}")
                return stats
                
        # Prepare column names for the output file
        col_names = ['chromosome', 'SNP_ID', 'bp', 'ref_allele', 'alt_allele', 'Global_SNP_Index']
        col_names.extend([col for col in file2_sample.columns])
    except Exception as e:
        print(f"Error reading file2 header or counting rows: {e}")
        return stats
    
    # Count total rows in file1
    try:
        print("Counting total SNPs in reference file")
        with open(file1_path, 'r') as f:
            for i, _ in enumerate(f):
                pass
        stats["file1_total_snps"] = i + 1  # +1 because i is 0-indexed
        print(f"Total SNPs in reference file: {stats['file1_total_snps']}")
    except Exception as e:
        print(f"Error counting reference file rows: {e}")
        # Continue anyway, this is just for reporting
    
    # If using threshold, count significant SNPs in file2 before chunked processing
    if use_threshold:
        print(f"Counting significant SNPs (p-value < {pvalue_threshold}) in disease file")
        chunksize = 5012000  # Use the same chunksize as for main processing
        file2_counter = pd.read_csv(file2_path, sep=r'\s+', chunksize=chunksize, low_memory=False)
        
        significant_count = 0
        for count_chunk in file2_counter:
            # Convert p-values to numeric
            count_chunk[pval_col] = pd.to_numeric(count_chunk[pval_col], errors='coerce')
            # Count SNPs below threshold
            significant_count += len(count_chunk[count_chunk[pval_col] < pvalue_threshold])
        
        stats["file2_significant_snps"] = significant_count
        print(f"Found {stats['file2_significant_snps']:,} significant SNPs in disease file")
    
    # Process reference file in smaller chunks to reduce memory usage
    chunksize = 5012000
    
    # To track which SNPs from file1 are matched
    matched_snp_indices = set()
    
    # Read file1 in chunks with low_memory=False to avoid dtype warnings
    file1_reader = pd.read_csv(file1_path, sep=r'\s+', header=None, 
                             names=['chromosome', 'SNP_ID', 'bp', 'ref_allele', 'alt_allele'],
                             chunksize=chunksize, low_memory=False)
    
    # Initialize output file - first write the header
    with open(output_path, 'w') as outfile:
        # Write header as CSV
        header = ','.join(col_names)
        outfile.write(f"{header}\n")
    
    # Flag to track if this is the first chunk (for CSV header)
    first_chunk = True
        
    for chunk_idx, file1_chunk in enumerate(file1_reader):
        print(f"Processing chunk {chunk_idx+1} of reference file:")
        
        # Add SNP_Index column (enumeration starting from chunk_idx*chunksize)
        base_index = chunk_idx * chunksize
        file1_chunk['SNP_Index'] = np.arange(base_index, base_index + len(file1_chunk))
        
        # Convert column types to ensure proper matching
        file1_chunk['chromosome'] = file1_chunk['chromosome'].astype(str)
        file1_chunk['bp'] = file1_chunk['bp'].astype(np.int32)  # Use more memory-efficient int32
        
        # Standardize allele case (convert all to uppercase)
        file1_chunk['ref_allele_upper'] = file1_chunk['ref_allele'].str.upper()
        file1_chunk['alt_allele_upper'] = file1_chunk['alt_allele'].str.upper()
        
        # Now process file2 in chunks against this chunk of file1, with low_memory=False
        file2_reader = pd.read_csv(file2_path, sep=r'\s+', chunksize=chunksize, low_memory=False)
        
        chunk_matches = 0
        chunk_matched_indices = set()
        
        # List to collect matches for this chunk
        matches_list = []
        
        for file2_chunk in file2_reader:
            original_chunk_size = len(file2_chunk)
            
            # Apply p-value filtering if thresholding is enabled
            if use_threshold:
                # Convert p-values to numeric and filter by threshold
                file2_chunk[pval_col] = pd.to_numeric(file2_chunk[pval_col], errors='coerce')
                
                # Apply p-value filter
                file2_chunk = file2_chunk[file2_chunk[pval_col] < pvalue_threshold].copy()
                
                # If no SNPs passed the filter, continue to the next chunk
                if len(file2_chunk) == 0:
                    continue
            
            # Convert column types for matching
            file2_chunk[chr_col] = file2_chunk[chr_col].astype(str)
            
            # Handle potential non-numeric position values
            try:
                file2_chunk[pos_col] = pd.to_numeric(file2_chunk[pos_col], errors='coerce')
                file2_chunk = file2_chunk.dropna(subset=[pos_col])  # Drop rows with NaN positions
                file2_chunk[pos_col] = file2_chunk[pos_col].astype(np.int32)
            except Exception as e:
                print(f"Warning: Error converting position column to numeric: {e}")
                print(f"Sample position values: {file2_chunk[pos_col].head()}")
                continue
            
            # Create allele uppercase versions for matching
            file2_chunk['allele1_upper'] = file2_chunk[allele1_col].str.upper()
            file2_chunk['allele2_upper'] = file2_chunk[allele2_col].str.upper()
            
            # Try direct matching
            merged_df = pd.merge(
                file1_chunk, 
                file2_chunk,
                left_on=['chromosome', 'bp', 'ref_allele_upper', 'alt_allele_upper'],
                right_on=[chr_col, pos_col, 'allele1_upper', 'allele2_upper'],
                how='inner'
            )
            
            # If no matches, try with alleles swapped
            if len(merged_df) == 0:
                merged_df = pd.merge(
                    file1_chunk, 
                    file2_chunk,
                    left_on=['chromosome', 'bp', 'ref_allele_upper', 'alt_allele_upper'],
                    right_on=[chr_col, pos_col, 'allele2_upper', 'allele1_upper'],
                    how='inner'
                )
            
            # If we have matches, collect them for writing
            if len(merged_df) > 0:
                # Remove duplicate SNPs based on SNP_Index, keeping only the first occurrence
                # This ensures each reference SNP is only included once in the output
                if 'SNP_Index' in merged_df.columns and len(merged_df) > len(merged_df['SNP_Index'].unique()):
                    print(f"    Found {len(merged_df) - len(merged_df['SNP_Index'].unique())} duplicate SNP indices - keeping first occurrences only")
                    merged_df = merged_df.drop_duplicates(subset=['SNP_Index'], keep='first')
                
                # Track matched SNP indices from file1
                chunk_matched_indices.update(merged_df['SNP_Index'].tolist())
                
                # Clean up temporary columns used for matching
                merged_df = merged_df.drop(['ref_allele_upper', 'alt_allele_upper', 
                                        'allele1_upper', 'allele2_upper'], axis=1)
                
                # Rename SNP_Index to Global_SNP_Index for the output
                merged_df = merged_df.rename(columns={'SNP_Index': 'Global_SNP_Index'})
                
                # Add to our list of matches (avoid concat with empty dataframes)
                matches_list.append(merged_df[col_names])
                
                chunk_matches += len(merged_df)
                
                # Clear merged_df to free memory
                del merged_df
                gc.collect()
            
            # Clear file2_chunk to free memory
            del file2_chunk
            gc.collect()
        
        # Write all matches for this chunk to CSV file
        if matches_list:  # Only if we have matches to write
            # Create dataframe from all matches collected in this chunk
            all_matches_df = pd.concat(matches_list, ignore_index=True) if len(matches_list) > 0 else pd.DataFrame()
            
            # Final check for duplicates across all merged chunks in this iteration
            if not all_matches_df.empty and 'Global_SNP_Index' in all_matches_df.columns:
                before_dedup = len(all_matches_df)
                all_matches_df = all_matches_df.drop_duplicates(subset=['Global_SNP_Index'], keep='first')
                after_dedup = len(all_matches_df)
                if before_dedup > after_dedup:
                    print(f"    Removed {before_dedup - after_dedup} duplicate Global_SNP_Index entries from final chunk output")
                    # Adjust match count if we removed duplicates
                    chunk_matches -= (before_dedup - after_dedup)
            
            # Write to CSV (mode='a' for append)
            if not all_matches_df.empty:
                all_matches_df.to_csv(output_path, mode='a', header=False, index=False)
                
            # Clear all_matches_df
            del all_matches_df
        
        # Update total overlapping SNPs count
        stats["overlapping_snps"] += chunk_matches
        
        # Update the master set of matched indices
        matched_snp_indices.update(chunk_matched_indices)
        
        print(f"  Chunk {chunk_idx+1}: {chunk_matches} matching SNPs")
        print(f"  Running total: {stats['overlapping_snps']} matching SNPs")
        
        # Clear matches_list and file1_chunk to free memory
        del matches_list
        del file1_chunk
        gc.collect()
    
    # Calculate statistics
    if use_threshold:
        stats["file2_nonoverlapping_snps"] = stats["file2_significant_snps"] - stats["overlapping_snps"]
        stats["filtered_snps"] = stats["file2_total_snps"] - stats["file2_significant_snps"]
    else:
        stats["file2_nonoverlapping_snps"] = stats["file2_total_snps"] - stats["overlapping_snps"]
    
    stats["file1_nonoverlapping_snps"] = stats["file1_total_snps"] - len(matched_snp_indices)
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    stats["processing_time_seconds"] = elapsed_time
    stats["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Print detailed summary statistics
    print("\n" + "=" * 70)
    print(f"DETAILED SUMMARY FOR {disease_name}")
    print("=" * 70)
    print(f"Reference file (file1) total SNPs: {stats['file1_total_snps']:,}")
    print(f"Disease file (file2) total SNPs: {stats['file2_total_snps']:,}")
    
    if use_threshold:
        print("\nP-VALUE FILTERING STATISTICS:")
        print(f"SNPs with p-value ≤ {pvalue_threshold}: {stats['file2_significant_snps']:,}")
        print(f"SNPs filtered out (p-value > {pvalue_threshold}): {stats['filtered_snps']:,}")
    
    print("\nOVERLAP STATISTICS:")
    print(f"Overlapping SNPs (match between files): {stats['overlapping_snps']:,}")
    print(f"Non-overlapping SNPs from disease file: {stats['file2_nonoverlapping_snps']:,}")
    print(f"Non-overlapping SNPs from reference file: {stats['file1_nonoverlapping_snps']:,}")
    
    print("- Processing complete!")
    
    # Write statistics to the stats file if provided
    if stats_file:
        # Check if the file exists to determine if we need to write headers
        file_exists = os.path.isfile(stats_file)
        stats_df = pd.DataFrame([stats])
        
        if file_exists:
            # Append without headers
            stats_df.to_csv(stats_file, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            stats_df.to_csv(stats_file, index=False)
    
    return stats

def extract_snps_for_multiple_diseases(use_threshold=False, pvalue_threshold=0.1):
    """
    Process GWAS data for multiple diseases, finding overlaps between the reference file
    and each disease-specific file using memory-optimized approach.
    
    Args:
        use_threshold: Whether to apply p-value thresholding (default: False)
        pvalue_threshold: P-value threshold if use_threshold=True (default: 0.1)
    """
    # Common reference file path
    file1_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen"
    
    # Base output directory
    output_dir = "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/extracted_5M"
    
    # Create a statistics file path based on threshold setting
    if use_threshold:
        stats_file = os.path.join(output_dir, f"processing_statistics_{pvalue_threshold}.csv")
        suffix = f"_{pvalue_threshold}"
    else:
        stats_file = os.path.join(output_dir, "processing_statistics_full.csv")
        suffix = "_full"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the diseases with their file paths and column mappings
    diseases = [
        {
            "name": "T2D",
            "file2_path": "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/Mahajan.NatGenet2018b.T2D.European.txt",
            "output_path": os.path.join(output_dir, f"t2d{suffix}.csv"),
            "column_mappings": {
                "chromosome": "Chr",
                "position": "Pos",
                "allele1": "EA",
                "allele2": "NEA",
                "pvalue": "Pvalue"
            }
        },
        {
            "name": "Prostate",
            "file2_path": "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/meta_v3_onco_euro_overall_ChrAll_1_release.txt",
            "output_path": os.path.join(output_dir, f"pros{suffix}.csv"),
            "column_mappings": {
                "chromosome": "Chr",
                "position": "position",
                "allele1": "Allele1",
                "allele2": "Allele2",
                "pvalue": "Pvalue" 
            }
        },
        {
            "name": "Pancreatic",
            "file2_path": "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/model1_ukb_imp_chr1-22_panc_merged.txt",
            "output_path": os.path.join(output_dir, f"pan{suffix}.csv"),
            "column_mappings": {
                "chromosome": "CHR",
                "position": "BP",
                "allele1": "ALLELE1",
                "allele2": "ALLELE0",
                "pvalue": "P_LINREG" 
            }
        },
        {
            "name": "Colon",
            "file2_path": "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/joint_wald_noUKB_MAC50_1_rsID.TBL",
            "output_path": os.path.join(output_dir, f"col{suffix}.csv"),
            "column_mappings": {
                "chromosome": "CHR",
                "position": "POS",
                "allele1": "Allele1",
                "allele2": "Allele2",
                "pvalue": "P.value" 
            }
        },
        {
            "name": "Breast",
            "file2_path": "/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/bcac_meta_rs.txt",
            "output_path": os.path.join(output_dir, f"brea{suffix}.csv"),
            "column_mappings": {
                "chromosome": "chr",
                "position": "position_b37",
                "allele1": "a0",
                "allele2": "a1",
                "pvalue": "bcac_onco_icogs_gwas_P1df" 
            }
        }
    ]
    
    # Process each disease and collect statistics
    all_stats = []
    for disease in diseases:
        try:
            stats = process_disease_data(
                disease["name"],
                file1_path,
                disease["file2_path"],
                disease["output_path"],
                disease["column_mappings"],
                use_threshold=use_threshold,
                pvalue_threshold=pvalue_threshold,
                stats_file=stats_file
            )
            all_stats.append(stats)
            # Force garbage collection between diseases
            gc.collect()
        except Exception as e:
            print(f"Error processing {disease['name']}: {e}")
    
    # Print overall summary
    print("\n\n" + "=" * 100)
    print("OVERALL PROCESSING SUMMARY".center(100))
    print("=" * 100)
    
    # Choose column headers based on threshold setting
    if use_threshold:
        print(f"{'Disease':<12} {'Ref SNPs':<10} {'Dis SNPs':<10} {'Sig SNPs':<10} {'Filtered':<10} {'Overlap':<10} {'Dis Non-Ovr':<12} {'Ref Non-Ovr':<12}")
    else:
        print(f"{'Disease':<12} {'Ref SNPs':<10} {'Dis SNPs':<10} {'Overlap':<10} {'Dis Non-Ovr':<12} {'Ref Non-Ovr':<12}")
    
    print("-" * 100)
    
    # Print summary rows for each disease
    for stats in all_stats:
        if use_threshold:
            print(f"{stats['disease_name']:<12} {stats['file1_total_snps']:<10,} {stats['file2_total_snps']:<10,} "
                  f"{stats['file2_significant_snps']:<10,} {stats['filtered_snps']:<10,} {stats['overlapping_snps']:<10,} "
                  f"{stats['file2_nonoverlapping_snps']:<12,} {stats['file1_nonoverlapping_snps']:<12,}")
        else:
            print(f"{stats['disease_name']:<12} {stats['file1_total_snps']:<10,} {stats['file2_total_snps']:<10,} "
                  f"{stats['overlapping_snps']:<10,} {stats['file2_nonoverlapping_snps']:<12,} "
                  f"{stats['file1_nonoverlapping_snps']:<12,}")
    
    print("=" * 100)
    print(f"Detailed statistics saved to: {stats_file}")
    print("Legend:")
    print("  - Ref SNPs: Total SNPs in reference file")
    print("  - Dis SNPs: Total SNPs in disease file")
    if use_threshold:
        print(f"  - Sig SNPs: SNPs that passed p-value threshold of {pvalue_threshold}")
        print("  - Filtered: SNPs filtered due to p-value threshold")
    print("  - Overlap: SNPs that match between both files")
    print("  - Dis Non-Ovr: " + ("Significant d" if use_threshold else "D") + "isease SNPs without a match in reference file")
    print("  - Ref Non-Ovr: Reference SNPs without a match in disease file")

if __name__ == "__main__":
    # Add requirements check
    try:
        import psutil
    except ImportError:
        print("psutil not installed. Installing...")
        import pip
        pip.main(['install', 'psutil'])
        import psutil
    
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process GWAS data for multiple diseases.')
    parser.add_argument('-use_threshold', action='store_true', 
                        help='Enable p-value thresholding (default: False)')
    parser.add_argument('-threshold', type=float, default=0.01,
                        help='P-value threshold to use if thresholding is enabled (default: 0.1)')
    
    args = parser.parse_args()
    
    # Run the function with the provided arguments
    extract_snps_for_multiple_diseases(use_threshold=args.use_threshold, 
                                     pvalue_threshold=args.threshold)