import os
import gzip
import glob
import shutil
import argparse
from multiprocessing import Pool
from pathlib import Path
import re
from itertools import islice
import gc

def natural_sort_key(s):
    """Function to sort strings with numbers in natural order"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_group_number(filename):
    """Extract the group number from filename (the number after the underscore)"""
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else None

def concatenate_group_files(args):
    """
    Concatenate all files for a specific chromosome and group number
    args: tuple of (chrom_num, group_num, input_folder, output_folder, lines_per_chunk)
    """
    try:
        chrom_num, group_num, input_folder, output_folder, lines_per_chunk = args
        
        # Create output directory if it doesn't exist
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Pattern to match files for this chromosome and group
        input_pattern = os.path.join(input_folder, f"chr{chrom_num}_{group_num}*.gen.gz")
        output_file = output_folder / f"chr{chrom_num}_{group_num}.gen.gz"
        
        # Get all matching files and sort them naturally
        matching_files = glob.glob(input_pattern)
        matching_files.sort(key=natural_sort_key)
        
        if not matching_files:
            print(f"No files found for chromosome {chrom_num}, group {group_num}")
            return
        
        print(f"Processing chromosome {chrom_num}, group {group_num}")
        print(f"Found {len(matching_files)} files to concatenate")
        
        total_lines = 0
        
        # Open output file
        with gzip.open(output_file, 'wt') as outfile:
            # Process each input file
            for i, infile_path in enumerate(matching_files, 1):
                print(f"  Processing file {i}/{len(matching_files)}: {os.path.basename(infile_path)}")

                file_lines = 0
                with gzip.open(infile_path, 'rt') as infile:
                    while True:
                        # Read chunk of lines
                        chunk = list(islice(infile, lines_per_chunk))
                        if not chunk:
                            break
                        
                        # Update line counts
                        chunk_lines = len(chunk)
                        file_lines += chunk_lines
                        
                        # Write chunk directly to output file
                        outfile.writelines(chunk)

                        # Free memory
                        del chunk
                        gc.collect()
                        
                        # Progress update for large files
                        if file_lines % (lines_per_chunk) == 0:
                            print(f" Processed {file_lines:,} lines - {os.path.basename(infile_path)}")
                
                total_lines += file_lines
                print(f"Chr{chrom_num}_group{group_num}: Completed {os.path.basename(infile_path)}: {file_lines:,} lines")
        
        print(f"Completed chromosome {chrom_num}, group {group_num}: {output_file}")
        print(f"Total lines in concatenated file: {total_lines:,}")
        return f"Chromosome {chrom_num}, group {group_num} completed with {total_lines:,} lines"

    except Exception as e:
        print(f"Error processing chromosome {chrom_num}, group {group_num}: {str(e)}")
        return f"Chromosome {chrom_num}, group {group_num} failed: {str(e)}"

def get_unique_groups(input_folder, chrom_num):
    """Get unique group numbers for a chromosome"""
    pattern = os.path.join(input_folder, f"chr{chrom_num}_*.gen.gz")
    files = glob.glob(pattern)
    groups = set()
    for f in files:
        group_num = get_group_number(os.path.basename(f))
        if group_num is not None:
            groups.add(group_num)
    return sorted(list(groups))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Concatenate chromosome files using multiprocessing')
    parser.add_argument('-input_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M', help='Input directory containing .gen.gz files')
    parser.add_argument('-output_folder',  type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_merged',help='Output directory for concatenated files')
    parser.add_argument('-threads',  type=int, default=12, help='Number of threads to use (default: number of CPU cores)')
    parser.add_argument('-lines', type=int, default=1000, help='Number of lines to process in each chunk (default: 1000)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input directory '{args.input_folder}' does not exist")
    
    print(f"Starting concatenation process:")
    print(f"Input directory: {args.input_folder}")
    print(f"Output directory: {args.output_folder}")
    
    # Create list of arguments for each process
    process_args = []
    for chrom in range(1, 23):  # Chromosomes 1-22
        groups = get_unique_groups(args.input_folder, chrom)
        for group in groups:
            process_args.append((chrom, group, args.input_folder, args.output_folder, args.lines))
    
    # Create a pool of workers
    with Pool(processes=args.threads) as pool:
        # Map the concatenation function to each chromosome-group combination
        results = pool.map(concatenate_group_files, process_args)
    
    # Print final summary
    print("\nProcessing Summary:")
    for result in results:
        if result:
            print(result)
    
    print(f"\nAll chromosomes and groups have been processed!")
    print(f"Concatenated files are available in: {args.output_folder}")

if __name__ == "__main__":
    main()