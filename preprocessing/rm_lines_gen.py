import gzip
import os
from pathlib import Path
import multiprocessing as mp

def process_file(file_info):
    input_file, output_file = file_info
    with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
        # Read and write the first 6807476 lines unchanged
        for _ in range(6807476):
            outfile.write(infile.readline())
        
        # Skip 30881 lines
        for _ in range(30881):
            infile.readline()
        
        # Write the remaining lines
        for line in infile:
            outfile.write(line)
    
    print(f"Finished processing {input_file.name}")

def main():
    input_folder = Path('/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_6_5files')  # Replace with your actual input folder path
    output_folder = Path('/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_6_processed')  # Replace with your desired output folder path
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Prepare list of file pairs to process
    file_pairs = []
    for i in range(17501, 21001):
        input_file = input_folder / f'sample_{i}.gen.gz'
        output_file = output_folder / f'sample_{i}.gen.gz'
        
        if input_file.exists():
            file_pairs.append((input_file, output_file))
        else:
            print(f"File {input_file.name} not found. Skipping.")
    
    # Use multiprocessing to process files
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_file, file_pairs)

if __name__ == '__main__':
    main()