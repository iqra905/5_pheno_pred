#************************************** Filter SNPs between the all_snps_chr_merged_maf_snps_0.15_t2d_0.1.csv gwas files and t2d_0.1_LDpruned.csv file ********************************
import pandas as pd
import os
import csv
from pathlib import Path

def main():
    # File paths
    file1_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered_plink_files/t2d/t2d_0.1_LDpruned.csv"
    file2_path = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new_study_split_20/t2d/all_snps_chr_merged_maf_snps_0.15_t2d_0.1.csv"
    output_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered_plink_files/t2d/t2d_0.1_LDpruned_maf_snps_0.15.csv"

    # Print file information
    print("File 1 Information:")
    file1_info = get_file_info(file1_path)
    for key, value in file1_info.items():
        print(f"  {key}: {value}")

    print("\nFile 2 Information:")
    file2_info = get_file_info(file2_path)
    for key, value in file2_info.items():
        print(f"  {key}: {value}")

    # Check if files exist
    if not file1_info.get("Exists", False) or not file2_info.get("Exists", False):
        missing_files = []
        if not file1_info.get("Exists", False):
            missing_files.append("File 1")
        if not file2_info.get("Exists", False):
            missing_files.append("File 2")
        print(f"Error: {', '.join(missing_files)} not found.")
        return

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect delimiters
        file1_delimiter = detect_delimiter(file1_path)
        file2_delimiter = detect_delimiter(file2_path)
        
        print(f"\nFile 1 delimiter detected as: '{file1_delimiter}'")
        print(f"File 2 delimiter detected as: '{file2_delimiter}'")
        
        # Read the header and check columns
        df1_header = pd.read_csv(file1_path, delimiter=file1_delimiter, nrows=0)
        df2_header = pd.read_csv(file2_path, delimiter=file2_delimiter, nrows=0)
        
        print("\nFile 1 columns:", df1_header.columns.tolist())
        print("File 2 columns:", df2_header.columns.tolist())
        
        # Check for the required columns
        if "CHR" not in df1_header.columns or "Global_SNP_Index" not in df1_header.columns:
            print("\nError: File 1 does not have the required columns 'CHR' and/or 'Global_SNP_Index'.")
            print("Available columns in File 1:", df1_header.columns.tolist())
            return
            
        if "Chromosome" not in df2_header.columns or "SNP_index_all" not in df2_header.columns:
            print("\nError: File 2 does not have the required columns 'Chromosome' and/or 'SNP_index_all'.")
            print("Available columns in File 2:", df2_header.columns.tolist())
            return
        
        # Count rows in File 1 for validation later
        file1_row_count = count_rows(file1_path, file1_delimiter)
        print(f"\nFile 1 has {file1_row_count} rows (excluding header)")
        
        # Process the files based on their size
        process_files(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count)
        
        # Print output file information
        print("\nOutput File Information:")
        output_info = get_file_info(output_path)
        for key, value in output_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def count_rows(file_path, delimiter):
    """Count the number of rows in a CSV file excluding the header."""
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            # Skip header
            next(reader, None)
            # Count rows
            return sum(1 for _ in reader)
    except Exception as e:
        print(f"Warning: Could not count rows in {file_path}: {str(e)}")
        return 0

def get_file_info(file_path):
    """Get information about a file."""
    path_obj = Path(file_path)
    if path_obj.exists():
        file_size = path_obj.stat().st_size / (1024 * 1024)  # Size in MB
        
        # Count lines (up to 5 for preview)
        preview_lines = []
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Only get first 5 lines for preview
                        preview_lines.append(line.strip())
                    else:
                        break
        except Exception as e:
            print(f"Warning: Could not read lines from {file_path}: {str(e)}")
            
        return {
            "Path": file_path,
            "Size (MB)": round(file_size, 2),
            "Exists": True,
            "Preview (first few lines)": preview_lines if preview_lines else []
        }
    else:
        return {
            "Path": file_path,
            "Exists": False
        }

def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file."""
    try:
        with open(file_path, 'r', newline='') as csvfile:
            sample = csvfile.read(4096)  # Read a sample
            
            # Try to detect using CSV Sniffer
            try:
                dialect = csv.Sniffer().sniff(sample)
                return dialect.delimiter
            except:
                # If sniffer fails, check for common delimiters
                delimiters = [',', '\t', ';', '|']
                counts = {d: sample.count(d) for d in delimiters}
                
                # Get the delimiter with the highest count
                if any(counts.values()):
                    max_delimiter = max(counts.items(), key=lambda x: x[1])[0]
                    return max_delimiter
                
                # Default to comma if no delimiter found
                return ','
                
    except Exception as e:
        print(f"Warning: Could not detect delimiter for {file_path}: {str(e)}")
        return ','

def process_files(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count):
    """Process the files and merge them."""
    # Determine which approach to use based on file sizes
    file1_size = os.path.getsize(file1_path) / (1024 * 1024)  # Size in MB
    file2_size = os.path.getsize(file2_path) / (1024 * 1024)  # Size in MB
    
    print(f"\nFile 1 size: {file1_size:.2f} MB")
    print(f"File 2 size: {file2_size:.2f} MB")
    
    
    print("\nBoth files can fit in memory. Processing them together...")
    process_in_memory(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count)


def process_in_memory(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count):
    """Process both files in memory."""
    print("Reading File 1...")
    df1 = pd.read_csv(file1_path, delimiter=file1_delimiter)
    print(f"File 1 shape: {df1.shape}")
    
    print("\nReading File 2...")
    df2 = pd.read_csv(file2_path, delimiter=file2_delimiter)
    print(f"File 2 shape: {df2.shape}")
    
    print("\nMerging files...")
    # Use suffixes to differentiate columns with the same name
    suffixes = ('_file1', '_file2')
    merged_df = pd.merge(
        df1, df2,
        left_on=["CHR", "Global_SNP_Index"],
        right_on=["Chromosome", "SNP_index_all"],
        suffixes=suffixes
    )
    
    print(f"Merged data shape: {merged_df.shape}")
      
    if not merged_df.empty:
        print("Merged data first few rows:")
        print(merged_df.head())
    else:
        print("No matches found between the files.")
    
    # Save the merged dataframe to a new CSV file
    print(f"\nSaving merged data to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    print(f"Merge completed. {len(merged_df)} rows written to output file.")

if __name__ == "__main__":
    main()
    
#************************************** Filter SNPs between the all_snps_chr_merged.csv gwas files and maf_snps_0.15.csv file ********************************
# import pandas as pd
# import os
# import csv
# from pathlib import Path

# def main():
#     # File paths
#     file1_path = "/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise/maf_filtered/maf_snps_0.15.csv"
#     file2_path = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new_study/t2d/all_snps_chr_merged.csv"
#     output_path = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new_study/t2d/all_snps_chr_merged_maf_snps_0.15_t2d.csv"

#     # Print file information
#     print("File 1 Information:")
#     file1_info = get_file_info(file1_path)
#     for key, value in file1_info.items():
#         print(f"  {key}: {value}")

#     print("\nFile 2 Information:")
#     file2_info = get_file_info(file2_path)
#     for key, value in file2_info.items():
#         print(f"  {key}: {value}")

#     # Check if files exist
#     if not file1_info.get("Exists", False) or not file2_info.get("Exists", False):
#         missing_files = []
#         if not file1_info.get("Exists", False):
#             missing_files.append("File 1")
#         if not file2_info.get("Exists", False):
#             missing_files.append("File 2")
#         print(f"Error: {', '.join(missing_files)} not found.")
#         return

#     try:
#         # Ensure output directory exists
#         output_dir = os.path.dirname(output_path)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Detect delimiters
#         file1_delimiter = detect_delimiter(file1_path)
#         file2_delimiter = detect_delimiter(file2_path)
        
#         print(f"\nFile 1 delimiter detected as: '{file1_delimiter}'")
#         print(f"File 2 delimiter detected as: '{file2_delimiter}'")
        
#         # Read the header and check columns
#         df1_header = pd.read_csv(file1_path, delimiter=file1_delimiter, nrows=0)
#         df2_header = pd.read_csv(file2_path, delimiter=file2_delimiter, nrows=0)
        
#         print("\nFile 1 columns:", df1_header.columns.tolist())
#         print("File 2 columns:", df2_header.columns.tolist())
        
#         # Check for the required columns
#         if "Chromosome" not in df1_header.columns or "SNP_Index" not in df1_header.columns:
#             print("\nError: File 1 does not have the required columns 'Chromosome' and/or 'SNP_Index'.")
#             print("Available columns in File 1:", df1_header.columns.tolist())
#             return
            
#         if "Chromosome" not in df2_header.columns or "SNP_Index" not in df2_header.columns:
#             print("\nError: File 2 does not have the required columns 'Chromosome' and/or 'SNP_Index'.")
#             print("Available columns in File 2:", df2_header.columns.tolist())
#             return
        
#         # Count rows in File 1 for validation later
#         file1_row_count = count_rows(file1_path, file1_delimiter)
#         print(f"\nFile 1 has {file1_row_count} rows (excluding header)")
        
#         # Process the files based on their size
#         process_files(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count)
        
#         # Print output file information
#         print("\nOutput File Information:")
#         output_info = get_file_info(output_path)
#         for key, value in output_info.items():
#             print(f"  {key}: {value}")
            
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()

# def count_rows(file_path, delimiter):
#     """Count the number of rows in a CSV file excluding the header."""
#     try:
#         with open(file_path, 'r') as f:
#             reader = csv.reader(f, delimiter=delimiter)
#             # Skip header
#             next(reader, None)
#             # Count rows
#             return sum(1 for _ in reader)
#     except Exception as e:
#         print(f"Warning: Could not count rows in {file_path}: {str(e)}")
#         return 0

# def get_file_info(file_path):
#     """Get information about a file."""
#     path_obj = Path(file_path)
#     if path_obj.exists():
#         file_size = path_obj.stat().st_size / (1024 * 1024)  # Size in MB
        
#         # Count lines (up to 5 for preview)
#         preview_lines = []
#         try:
#             with open(file_path, 'r') as f:
#                 for i, line in enumerate(f):
#                     if i < 5:  # Only get first 5 lines for preview
#                         preview_lines.append(line.strip())
#                     else:
#                         break
#         except Exception as e:
#             print(f"Warning: Could not read lines from {file_path}: {str(e)}")
            
#         return {
#             "Path": file_path,
#             "Size (MB)": round(file_size, 2),
#             "Exists": True,
#             "Preview (first few lines)": preview_lines if preview_lines else []
#         }
#     else:
#         return {
#             "Path": file_path,
#             "Exists": False
#         }

# def detect_delimiter(file_path):
#     """Detect the delimiter used in a CSV file."""
#     try:
#         with open(file_path, 'r', newline='') as csvfile:
#             sample = csvfile.read(4096)  # Read a sample
            
#             # Try to detect using CSV Sniffer
#             try:
#                 dialect = csv.Sniffer().sniff(sample)
#                 return dialect.delimiter
#             except:
#                 # If sniffer fails, check for common delimiters
#                 delimiters = [',', '\t', ';', '|']
#                 counts = {d: sample.count(d) for d in delimiters}
                
#                 # Get the delimiter with the highest count
#                 if any(counts.values()):
#                     max_delimiter = max(counts.items(), key=lambda x: x[1])[0]
#                     return max_delimiter
                
#                 # Default to comma if no delimiter found
#                 return ','
                
#     except Exception as e:
#         print(f"Warning: Could not detect delimiter for {file_path}: {str(e)}")
#         return ','

# def process_files(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count):
#     """Process the files and merge them."""
#     # Determine which approach to use based on file sizes
#     file1_size = os.path.getsize(file1_path) / (1024 * 1024)  # Size in MB
#     file2_size = os.path.getsize(file2_path) / (1024 * 1024)  # Size in MB
    
#     print(f"\nFile 1 size: {file1_size:.2f} MB")
#     print(f"File 2 size: {file2_size:.2f} MB")
    
#     # Choose processing approach based on file sizes
#     if file1_size + file2_size < 1000:  # If combined size is less than 1GB
#         print("\nBoth files can fit in memory. Processing them together...")
#         process_in_memory(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count)
#     else:
#         print("\nFiles are large. Processing in chunks...")
#         process_with_chunks(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count)

# def process_in_memory(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count):
#     """Process both files in memory."""
#     print("Reading File 1...")
#     df1 = pd.read_csv(file1_path, delimiter=file1_delimiter)
#     print(f"File 1 shape: {df1.shape}")
    
#     print("\nReading File 2...")
#     df2 = pd.read_csv(file2_path, delimiter=file2_delimiter)
#     print(f"File 2 shape: {df2.shape}")
    
#     print("\nMerging files...")
#     # Use suffixes to differentiate columns with the same name
#     suffixes = ('_file1', '_file2')
#     merged_df = pd.merge(
#         df1, df2,
#         left_on=["Chromosome", "SNP_Index"],
#         right_on=["Chromosome", "SNP_Index"],
#         suffixes=suffixes
#     )
    
#     print(f"Merged data shape: {merged_df.shape}")
    
#     # Verify that all rows from File 1 have matches
#     if len(merged_df) != len(df1):
#         print(f"\nWARNING: Not all rows from File 1 were matched in File 2!")
#         print(f"File 1 row count: {len(df1)}")
#         print(f"Merged output row count: {len(merged_df)}")
#         print(f"Missing matches: {len(df1) - len(merged_df)} rows")
        
#         # Create a tracking set for matches
#         # We need to handle the right column naming due to suffixes
#         chromosome_col = "Chromosome" if "Chromosome" in merged_df.columns else "Chromosome_file1"
#         snp_index_col = "SNP_Index_file1" if "SNP_Index_file1" in merged_df.columns else "SNP_Index"
        
#         merged_keys = set(zip(merged_df[chromosome_col], merged_df[snp_index_col]))
        
#         # Find unmatched rows
#         unmatched_mask = ~df1.apply(lambda row: (row["Chromosome"], row["SNP_Index"]) in merged_keys, axis=1)
#         unmatched_rows = df1[unmatched_mask]
        
#         # Sample of unmatched rows
#         if not unmatched_rows.empty:
#             print(f"\nSample of unmatched rows (up to 5):")
#             print(unmatched_rows.head())
            
#             # Write unmatched rows to a separate file for further investigation
#             unmatched_file = output_path.replace(".csv", "_unmatched.csv")
#             unmatched_rows.to_csv(unmatched_file, index=False)
#             print(f"All unmatched rows written to: {unmatched_file}")
#     else:
#         print("\nSUCCESS: All rows from File 1 were successfully matched in File 2, as expected.")
    
#     if not merged_df.empty:
#         print("Merged data first few rows:")
#         print(merged_df.head())
#     else:
#         print("No matches found between the files.")
    
#     # Save the merged dataframe to a new CSV file
#     print(f"\nSaving merged data to {output_path}...")
#     merged_df.to_csv(output_path, index=False)
    
#     print(f"Merge completed. {len(merged_df)} rows written to output file.")
    
#     # Final verification message
#     if len(merged_df) == file1_row_count:
#         print("\nFINAL VERIFICATION: The output file contains exactly the same number of rows as File 1, confirming that File 1 is a subset of File 2.")
#     else:
#         print(f"\nFINAL VERIFICATION FAILED: Output has {len(merged_df)} rows but File 1 has {file1_row_count} rows.")
#         missing_percentage = ((file1_row_count - len(merged_df)) / file1_row_count) * 100
#         print(f"Missing matches: {missing_percentage:.2f}% of File 1 rows could not be found in File 2.")

# def process_with_chunks(file1_path, file2_path, output_path, file1_delimiter, file2_delimiter, file1_row_count):
#     """Process files using chunking approach for large files."""
#     # Since we need to verify all rows from File 1 match, we'll read File 1 completely
#     # But we'll process File 2 in chunks to save memory
#     print("Reading File 1 completely...")
#     df1 = pd.read_csv(file1_path, delimiter=file1_delimiter)
#     print(f"File 1 shape: {df1.shape}")
    
#     # Create a tracking DataFrame to mark which rows from File 1 have matches
#     df1['has_match'] = False
    
#     # Create a new CSV file for the merged output
#     chunk_size = 100000  # Adjust based on memory constraints
#     is_first_chunk = True
#     total_merged_rows = 0
    
#     print("Processing File 2 in chunks...")
#     for chunk_idx, chunk_df2 in enumerate(pd.read_csv(file2_path, delimiter=file2_delimiter, chunksize=chunk_size)):
#         print(f"Processing chunk {chunk_idx+1} of File 2 with {len(chunk_df2)} rows...")
        
#         # Merge this chunk with File 1
#         suffixes = ('_file1', '_file2')
#         chunk_merged = pd.merge(
#             df1[~df1['has_match']], # Only try to match rows that haven't been matched yet
#             chunk_df2,
#             left_on=["Chromosome", "SNP_Index"],
#             right_on=["Chromosome", "SNP_Index"],
#             suffixes=suffixes
#         )
        
#         if not chunk_merged.empty:
#             # Get the matched indices from df1
#             chromosome_col = "Chromosome" if "Chromosome" in chunk_merged.columns else "Chromosome_file1"
#             snp_index_col = "SNP_Index_file1" if "SNP_Index_file1" in chunk_merged.columns else "SNP_Index"
            
#             matched_keys = set(zip(chunk_merged[chromosome_col], chunk_merged[snp_index_col]))
            
#             # Mark rows in df1 as matched
#             for i, row in df1.iterrows():
#                 if (row["Chromosome"], row["SNP_Index"]) in matched_keys:
#                     df1.at[i, 'has_match'] = True
            
#             # Write to output file
#             chunk_merged_copy = chunk_merged.copy()
#             # Drop the has_match column before saving
#             if 'has_match' in chunk_merged_copy.columns:
#                 chunk_merged_copy = chunk_merged_copy.drop('has_match', axis=1)
                
#             chunk_merged_copy.to_csv(
#                 output_path, 
#                 mode='a' if not is_first_chunk else 'w',
#                 header=is_first_chunk,
#                 index=False
#             )
            
#             is_first_chunk = False
#             total_merged_rows += len(chunk_merged)
#             print(f"Merged {len(chunk_merged)} rows in this chunk. Total merged rows so far: {total_merged_rows}")
            
#             # If all rows from File 1 have been matched, we can stop processing
#             if df1['has_match'].all():
#                 print("All rows from File 1 have been matched. Stopping further processing.")
#                 break
    
#     # Check for unmatched rows
#     unmatched_rows = df1[~df1['has_match']]
#     if not unmatched_rows.empty:
#         print(f"\nWARNING: {len(unmatched_rows)} rows from File 1 were not matched in File 2!")
#         print(f"File 1 row count: {len(df1)}")
#         print(f"Merged output row count: {total_merged_rows}")
        
#         print(f"Sample of unmatched rows (up to 5):")
#         print(unmatched_rows.drop('has_match', axis=1).head())
        
#         # Write unmatched rows to a separate file for further investigation
#         unmatched_file = output_path.replace(".csv", "_unmatched.csv")
#         unmatched_rows.drop('has_match', axis=1).to_csv(unmatched_file, index=False)
#         print(f"All unmatched rows written to: {unmatched_file}")
#     else:
#         print("\nSUCCESS: All rows from File 1 were successfully matched in File 2, as expected.")
    
#     print(f"\nMerge completed. Total rows in merged file: {total_merged_rows}")
    
#     # Final verification message
#     if total_merged_rows == file1_row_count:
#         print("\nFINAL VERIFICATION: The output file contains exactly the same number of rows as File 1, confirming that File 1 is a subset of File 2.")
#     else:
#         print(f"\nFINAL VERIFICATION FAILED: Output has {total_merged_rows} rows but File 1 has {file1_row_count} rows.")
#         missing_percentage = ((file1_row_count - total_merged_rows) / file1_row_count) * 100
#         print(f"Missing matches: {missing_percentage:.2f}% of File 1 rows could not be found in File 2.")

# if __name__ == "__main__":
#     main()