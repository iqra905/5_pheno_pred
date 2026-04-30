import os
import gzip
import csv
import pandas as pd
from multiprocessing import Pool, cpu_count

def read_snp_indices(csv_file):
    snp_indices = set()
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snp_indices.add(int(row['global_index']))
            #snp_indices.add(int(row['bim_index']))
            #snp_indices.add(int(row['SNP_Index']))
            
    return snp_indices

def process_gz_file(args):
    input_file, output_file, snp_indices, num_columns = args
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
        for i, line in enumerate(in_f, 0):
            if i in snp_indices:
                out_f.write(line)
                
    return f"Processed {input_file} -> {output_file}"

def main():
    # Directory containing .gen.gz files
    brea_can_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can'
    
    # CSV file with SNP indices
    csv_file = '/vol/research/fmodal_mmmed/Codes/DeepCombi/tests/results/brea/top_67260_snps.csv'
    
    # First 5 columns file
    first_5_col_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen.gz'
    
    # New directories for updated files
    updated_brea_can_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_weights_deepcombi_2'
    #updated_first_5_brea_can_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/'
    
    # Read SNP indices from CSV
    snp_indices = read_snp_indices(csv_file)
    
    # Prepare arguments for multiprocessing
    args_list = []

    # Add first_5_col_col_gen.gz file
    input_file = first_5_col_file
    file_name, file_ext = os.path.splitext(first_5_col_file)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    output_file = f"{file_name}_weights_deepcombi_2.gen.gz"
    args_list.append((input_file, output_file, snp_indices, 5))
    
    # Add .gen.gz files from col directory
    for filename in os.listdir(brea_can_dir):
        if filename.endswith('.gen.gz'):
            input_file = os.path.join(brea_can_dir, filename)
            output_file = os.path.join(updated_brea_can_dir, filename)
            args_list.append((input_file, output_file, snp_indices, 3))
    
    
    # Use multiprocessing to process files
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_gz_file, args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
