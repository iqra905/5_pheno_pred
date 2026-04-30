import gzip
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='Extract per disease SNPs from 5M SNPs in sample files.')
parser.add_argument('-input_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pan_can', help='Path to the folder with per disease sample_n.gen.gz files')
parser.add_argument('-output_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pan_can_snps', help='Output folder')
parser.add_argument('-bim_files', type=str, default=['/vol/vssp/SF_ucdatasets/gwas/data_files/Archive/Panscan3_b37.bim','/vol/vssp/SF_ucdatasets/gwas/data_files/Archive/Panscan_b37.bim'], help='Path to .bim files')
parser.add_argument('-summary_file', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pan_can_snps/processing_summary_pan.txt', help='Path for the summary output file')
args = parser.parse_args()


def process_file(filename, input_folder, output_folder, bim_data):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    rows_read = 0
    rows_written = 0
    
    with gzip.open(input_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
        for line in in_file:
            rows_read += 1
            fields = line.strip().split()
            chro = int(fields[0])
            location = int(fields[2])
                       
            if (chro, location) in bim_data:
                out_file.write(line)
                rows_written += 1
    
    rows_dropped = rows_read - rows_written
    return filename, rows_read, rows_written, rows_dropped

def read_and_process_bim_files(bim_files):
    all_bim_data = pd.DataFrame()
    
    for bim_file in bim_files:
        if os.path.exists(bim_file):
            print(f"Reading BIM file: {bim_file}")
            bim_df = pd.read_csv(bim_file, sep=r'\s+', engine='python', header=None, 
                                 names=['chr', 'snp_id', 'distance', 'location', 'ref', 'alt'])
            all_bim_data = pd.concat([all_bim_data, bim_df])
        else:
            print(f"Warning: BIM file not found: {bim_file}")
    
    if all_bim_data.empty:
        print("Error: No valid BIM data found. Exiting.")
        return None
    
    print(f"Total number of SNPs from {len(bim_files)} chips: {len(all_bim_data)}")


    # Remove duplicates based on chr, location, ref, and alt
    all_bim_data.drop_duplicates(subset=['chr', 'location', 'ref', 'alt'], keep='first', inplace=True)
    
    print("\nCombined BIM data statistics:")
    print(f"Total number of unique SNPs in {len(bim_files)} chips: {len(all_bim_data)}")
    print(all_bim_data.head())

    bim_data = set(zip(all_bim_data['chr'], all_bim_data['location']))
    print(f"Number of unique (chr, location) tuples: {len(bim_data)}")
    
    return bim_data

def main():
    # Define paths
    input_folder = args.input_folder
    output_folder =args.output_folder
    bim_files = args.bim_files
    summary_file = args.summary_file

    os.makedirs(output_folder, exist_ok=True)

    # Read and process BIM files
    bim_data = read_and_process_bim_files(bim_files)
    if bim_data is None:
        return

    # Get list of .gen.gz files and sort them
    gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Set up multiprocessing
    num_processes = min(cpu_count(), len(gen_files))
    pool = Pool(processes=num_processes)

    # Process files in parallel
    print(f"Processing {len(gen_files)} files using {num_processes} processes...")
    process_func = partial(process_file, input_folder=input_folder, output_folder=output_folder, bim_data=bim_data)
    results = pool.map(process_func, gen_files)

    pool.close()
    pool.join()

    # Sort results
    results.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))

    # Print results
    print("\nProcessing complete. Results:")
    for filename, rows_read, rows_written, rows_dropped in results:
        print(f"{filename}:\n Rows read: {rows_read}: Rows written: {rows_written}: Rows dropped: {rows_dropped}\n")  

    # Write results to summary file
    with open(summary_file, 'w') as f:
        f.write(f"BIM files used:\n")
        for bim_file in bim_files:
            f.write(f"- {bim_file}\n")
        f.write(f"\nNumber of unique SNPs after removing duplicates: {len(bim_data)}\n")
        f.write(f"Number of files processed: {len(gen_files)}\n\n")
        f.write("File-wise statistics:\n")
        
        for filename, rows_read, rows_written, rows_dropped in results:
            f.write(f"{filename}:\n Rows read: {rows_read}: Rows written: {rows_written}: Rows dropped: {rows_dropped}\n")  
            
    print(f"\nProcessing complete. Summary written to {summary_file}")

if __name__ == '__main__':
    main()