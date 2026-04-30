import os
import gzip
import csv
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from filelock import FileLock
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('genotype_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def sort_csv_file(csv_file):
    csv_file = Path(csv_file)
    logger.info(f"Sorting CSV file: {csv_file}")
    
    try:
        # Create a temporary file for the sorted output
        temp_file = csv_file.with_suffix('.temp.csv')
        
        # Read the data from the CSV file
        data = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Get the header row
            
            # Find column indices for sorting
            try:
                chr_idx = header.index('Chromosome')
                snp_idx = header.index('SNP_Index')
            except ValueError:
                logger.error(f"Could not find Chromosome or SNP_Index columns in {csv_file}")
                return
                
            # Read all data into memory
            for row in reader:
                data.append(row)
        
        # Handle potential empty data case
        if not data:
            logger.warning(f"No data found in {csv_file} to sort")
            return
            
        def sort_key(row):
            try:
                chr_val = int(row[chr_idx]) if row[chr_idx] and row[chr_idx] != 'None' else float('inf')
                snp_val = int(row[snp_idx]) if row[snp_idx] and row[snp_idx] != 'None' else float('inf')
                return (chr_val, snp_val)
            except (ValueError, IndexError):
                logger.warning(f"Issue with sorting row: {row}")
                return (float('inf'), float('inf'))
                
        sorted_data = sorted(data, key=sort_key)
        
        # Write the sorted data to the temporary file
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted_data)
            
        # Replace the original file with the sorted file
        temp_file.replace(csv_file)
        logger.info(f"Successfully sorted {csv_file} by chromosome and SNP index")
        
    except Exception as e:
        logger.error(f"Error sorting CSV file {csv_file}: {e}", exc_info=True)

def custom_sort(file_name):
    if file_name.startswith('chr') and file_name.endswith('.gen.gz'):
        try:
            # Extract the number between 'chr' and '.gen.gz'
            chr_str = file_name[3:].split('.')[0]
            chr_num = int(chr_str)
            return (chr_num, file_name)
        except (ValueError, IndexError):
            pass
    
    # Fallback for other naming patterns
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def process_genotype_file(file_path, output_folder, summary_file, maf_threshold, 
                          kept_snps_csv, filtered_snps_csv):
    file_path = Path(file_path)
    base_name = file_path.name
    logger.info(f"Processing file: {base_name}")
    
    # Extract chromosome number from filename (chr1.gen.gz -> 1)
    chr_num = None
    if base_name.startswith('chr') and base_name.endswith('.gen.gz'):
        try:
            chr_str = base_name[3:].split('.')[0]
            chr_num = int(chr_str)
        except (ValueError, IndexError):
            logger.warning(f"Could not extract chromosome number from {base_name}")
    
    output_path = Path(output_folder) / base_name
    skipped_count = 0
    written_count = 0
    total_lines = 0
    snp_index = 0  

    # Create file locks to prevent concurrent writing issues
    summary_lock = FileLock(f"{summary_file}.lock")
    kept_lock = FileLock(f"{kept_snps_csv}.lock")
    filtered_lock = FileLock(f"{filtered_snps_csv}.lock")

    try:
        with gzip.open(file_path, 'rt') as f_in, gzip.open(output_path, 'wt') as f_out:
            # Open CSV files with locks for thread safety
            for line in f_in:
                total_lines += 1
                # Use current snp_index value (starting from 0) then increment for next SNP
                current_snp_index = snp_index
                snp_index += 1

                # Log progress for every 50000 SNPs
                if total_lines % 50000 == 0:
                    logger.info(f"Processing {base_name}: {total_lines} SNPs processed so far")
                
                columns = line.strip().split()
                
                # Store first 5 columns (SNP info)
                snp_info = columns[:5]
                
                # Skip if we don't have enough columns
                if len(columns) < 8:  # At least 5 + 3 for one individual
                    logger.warning(f"Line {total_lines} in {base_name} has insufficient columns")
                    skipped_count += 1
                    continue
                
                num_individuals = (len(columns) - 5) // 3
                
                # Calculate allele frequencies
                allele_counts = np.zeros(2)  # [allele1_count, allele2_count]
                total_alleles = 2 * num_individuals  # Each individual has 2 alleles
                
                try:
                    for i in range(num_individuals):
                        # Extract genotype probabilities: P(AA), P(AB), P(BB)
                        aa_prob = float(columns[5 + i*3])
                        ab_prob = float(columns[5 + i*3 + 1])
                        bb_prob = float(columns[5 + i*3 + 2])
                        
                        # Expected allele count contribution from this individual
                        allele_counts[0] += (2 * aa_prob + ab_prob)  # 2 copies if AA, 1 if AB
                        allele_counts[1] += (2 * bb_prob + ab_prob)  # 2 copies if BB, 1 if AB
                    
                    # Calculate frequencies
                    allele_freqs = allele_counts / total_alleles
                    maf = min(allele_freqs)
                    
                    # Filter based on MAF threshold
                    if maf < maf_threshold:
                        skipped_count += 1
                        
                        # Write filtered SNP info to CSV with SNP_Index
                        with filtered_lock:
                            with open(filtered_snps_csv, 'a', newline='') as f_filtered:
                                writer = csv.writer(f_filtered)
                                writer.writerow(snp_info + [maf, chr_num, current_snp_index])
                        continue
                        
                    # Write the SNP to the output file
                    f_out.write(line)
                    written_count += 1
                    
                    # Write kept SNP info to CSV with SNP_Index
                    with kept_lock:
                        with open(kept_snps_csv, 'a', newline='') as f_kept:
                            writer = csv.writer(f_kept)
                            writer.writerow(snp_info + [maf, chr_num, current_snp_index])
                
                except (ValueError, IndexError) as e:
                    logger.error(f"Error processing line {total_lines} in {base_name}: {e}")
                    skipped_count += 1
                    continue

        # Write summary information
        with summary_lock:
            with open(summary_file, 'a') as f:
                f.write(f"{base_name}: Total SNPs {total_lines}, Dropped {skipped_count}, Written {written_count}\n")

        logger.info(f"Processed file: {base_name}")
        logger.info(f"Total SNPs: {total_lines}, Dropped: {skipped_count}, Written: {written_count}")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path} - {e}")
    except Exception as e:
        logger.error(f"Error processing file: {file_path} - {e}", exc_info=True)

def main(input_folder, output_folder, start_file_index, end_file_index, 
         output_file, maf_threshold, kept_snps_csv, filtered_snps_csv):
    
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create CSV headers
    csv_header = ['CHR', 'RSID', 'Position', 'Allele1', 'Allele2', 'MAF', 'Chromosome', 'SNP_Index']
    
    # Initialize CSV files with headers
    for csv_file in [kept_snps_csv, filtered_snps_csv]:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    try:
        files = list(input_folder.glob('chr*.gen.gz'))
        if not files:
            logger.warning(f"No chr*.gen.gz files found in {input_folder}")
            # Fall back to any .gen.gz files
            files = list(input_folder.glob('*.gen.gz'))
            if not files:
                logger.error(f"No .gen.gz files found in {input_folder}")
                return
    except FileNotFoundError:
        logger.error(f"Error: Input folder '{input_folder}' not found.")
        return
    except PermissionError:
        logger.error(f"Error: Permission denied for accessing '{input_folder}'.")
        return

    sorted_files = sorted(files, key=lambda f: custom_sort(f.name))
    total_files = len(sorted_files)
    logger.info(f"Found {total_files} genotype files.")

    # Validate indices
    if start_file_index < 0 or start_file_index >= total_files:
        logger.error(f"Invalid start_index: {start_file_index}")
        return
    
    if end_file_index <= start_file_index or end_file_index > total_files:
        end_file_index = total_files
        logger.warning(f"Adjusted end_index to {end_file_index}")

    files_to_process = sorted_files[start_file_index:end_file_index]
    
    # Write processing information
    with open(output_file, 'w') as f:
        f.write(f"Processing started at: {datetime.now()}\n")
        f.write(f"MAF threshold: {maf_threshold}\n")
        f.write(f"Processing {len(files_to_process)} files (from file {start_file_index+1} to file {end_file_index})\n")
        f.write(f"File names under process are: {', '.join([file.name for file in files_to_process])}\n\n")

    # Process files in parallel
    with multiprocessing.Pool() as pool:
        pool.starmap(
            process_genotype_file, 
            [(file, output_folder, output_file, maf_threshold, kept_snps_csv, filtered_snps_csv) 
             for file in files_to_process]
        )

    # Calculate and write summary statistics
    total_snps = 0
    total_skipped = 0
    total_written = 0

    with open(output_file, 'r') as f:
        for line in f:
            if "Total SNPs" in line:
                try:
                    parts = line.split(',')
                    total_snps += int(parts[0].split()[-1])
                    total_skipped += int(parts[1].split()[-1])
                    total_written += int(parts[2].split()[-1])
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing summary line: {line} - {e}")

    with open(output_file, 'a') as f:
        f.write(f"\nSummary:\n")
        f.write(f"Total SNPs across all files: {total_snps}\n")
        f.write(f"Total SNPs skipped (MAF < {maf_threshold}): {total_skipped}\n")
        f.write(f"Total SNPs written: {total_written}\n")
        f.write(f"Processing completed at: {datetime.now()}\n\n")

    logger.info(f"Processing completed. Total SNPs: {total_snps}, Skipped: {total_skipped}, Written: {total_written}")
    
    # Sort CSV files by chromosome number and SNP index
    logger.info("Sorting CSV files by chromosome number and SNP index...")
    sort_csv_file(kept_snps_csv)
    sort_csv_file(filtered_snps_csv)
    
    logger.info(f"Summary written to {output_file}")
    logger.info(f"Kept SNPs info written to {kept_snps_csv}")
    logger.info(f"Filtered SNPs info written to {filtered_snps_csv}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process genotype files by filtering based on minor allele frequency.")
    parser.add_argument("-input_folder", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise', help="Path to the input folder containing .gen.gz files")
    parser.add_argument("-output_folder", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise/maf_filtered', help="Path to the output folder to save filtered files")
    parser.add_argument("-output_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise/maf_filtered/summary.txt', help="Path to the summary output file")
    parser.add_argument("-kept_snps_csv", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise/maf_filtered/maf_snps_0.15.csv', help="Path to CSV file for storing information about kept SNPs")
    parser.add_argument("-filtered_snps_csv", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise/maf_filtered/filtered_maf_snps_0.15.csv', help="Path to CSV file for storing information about filtered SNPs")
    parser.add_argument("-start_index", type=int, default=0, help="Index of the first file to process (0-based)")
    parser.add_argument("-end_index", type=int, default=None, help="Index of the last file to process (0-based), defaults to all files")
    parser.add_argument("-maf_threshold", type=float, default=0.15, help="Minor allele frequency threshold (SNPs with MAF below this value will be filtered out)")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.start_index, args.end_index or float('inf'), 
         args.output_file, args.maf_threshold, args.kept_snps_csv, args.filtered_snps_csv)


