import gzip
import os
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

parser = argparse.ArgumentParser(description='Extract 5M SNPs from 5-disease text file from sample files.')
parser.add_argument('-input_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_11', help='Path to the folder with sample_n.gen.gz files')
parser.add_argument('-output_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M', help='Output folder')
parser.add_argument('-txt_file', type=str, default='/vol/research/ucdatasets/gwas/data_files/5d_gwas_05maf_0001hwe_08info_VL2.txt', help='Path to .txt file')
parser.add_argument('-summary_file', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M/7M_5M_processing_summary.txt' , help='Path for the summary output file')
parser.add_argument('-slice', type=str, default='0:10000', help='Slice of files to process (e.g., "0:10000", "10000:20000", "20000:")')
args = parser.parse_args()

def process_file(filename, input_folder, output_folder, txt_data):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    rows_read = 0
    rows_written = 0
    
    with gzip.open(input_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
        for line in in_file:
            rows_read += 1
            fields = line.strip().split()
            chro = int(fields[0])  #  CHR is in the 1st column 
            location = int(fields[2])  # Assuming location is in the 3rd column 
            
            if (chro, location) in txt_data:
                out_file.write(line)  # Write the original line as it is
                rows_written += 1
    
    rows_dropped = rows_read - rows_written
    return filename, rows_read, rows_written, rows_dropped

def main():
    # Define paths
    input_folder = args.input_folder
    output_folder = args.output_folder
    txt_file = args.txt_file
    summary_file = args.summary_file
    slice_str = args.slice

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read .bim file
    print("Reading .txt file...")
    txt_df = pd.read_csv(txt_file, sep=r'\s+')
    print("TXT file read successfully.")
    print(f"Number of SNPs in TXT file: {len(txt_df)}")
    print(txt_df.head())

    # # Check for duplicate positions in txt file
    # txt_duplicates = txt_df[txt_df.duplicated(subset=['Chromosome', 'Position'], keep=False)]
    # print(f"Number of duplicate (chr, position) pairs in TXT file: {len(txt_duplicates)}")


    # Create a set of (chr, location) tuples from .txt file for faster lookup
    txt_data = set(zip(txt_df['Chromosome'], txt_df['Position']))
    print(f"\nNumber of unique (chr, position) tuples in TXT file: {len(txt_data)}")
    print("TXT data created successfully.")
    #print(type(txt_data))
    #print(len(txt_data))
    #print(bim_data)

    # Get list of .gen.gz files and sort them
    gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Parse the slice string
    slice_parts = slice_str.split(':')
    start = int(slice_parts[0]) if slice_parts[0] else None
    end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None

    # Select subset of files based on the slice
    subset_files = gen_files[start:end]

    print(f"Processing files {start or 0} to {end or 'end'}")
    print(f"Number of files to process: {len(subset_files)}")

    # Set up multiprocessing
    num_processes = min(cpu_count(), len(subset_files))
    pool = Pool(processes=num_processes)

    # Process files in parallel
    print(f"Processing {len(subset_files)} files using {num_processes} processes...")
    process_func = partial(process_file, input_folder=input_folder, output_folder=output_folder, txt_data=txt_data)
    results = pool.map(process_func, subset_files)

    # Close the pool
    pool.close()
    pool.join()

    # Sort results to match the order of processed files
    results.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))

    # Print results
    print("\nProcessing complete. Results:")
    for filename, rows_read, rows_written, rows_dropped in results:
        print(f"{filename}:\n SNPs read: {rows_read}: SNPs written: {rows_written}: SNPs dropped: {rows_dropped}\n")  

    print("\nAll files processed successfully.")


    # Write results to summary file
    with open(summary_file, 'w') as f:
        f.write(f"TXT file: {txt_file}\n")
        f.write(f"Number of files processed: {len(subset_files)}\n\n")
        f.write("File-wise statistics:\n")
        
        for filename, rows_read, rows_written, rows_dropped in results:
            f.write(f"{filename}:\n SNPs read: {rows_read}: SNPs written: {rows_written}: SNPs dropped: {rows_dropped}\n")  
            
    print(f"\nProcessing complete. Summary written to {summary_file}")

if __name__ == '__main__':
    main()

