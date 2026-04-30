import os
import gzip
import csv
import pandas as pd
from multiprocessing import Pool, cpu_count

# def read_snp_indices(csv_file):
#     snp_indices = set()
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             snp_indices.add(int(row['SNP_Index']))
#     return snp_indices

def read_snp_indices(csv_file):
    snp_indices = set()
    index_counts = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snp_index = int(row['SNP_Index'])
            
            # Count occurrences
            if snp_index in index_counts:
                index_counts[snp_index] += 1
            else:
                index_counts[snp_index] = 1
                
            snp_indices.add(snp_index)
    
    # Find duplicates
    duplicates = {idx: count for idx, count in index_counts.items() if count > 1}
    print(f"Total rows in CSV: {sum(index_counts.values())}")
    print(f"Unique SNP indices added: {len(snp_indices)}")
    print(f"Duplicate indices: {duplicates}")
    
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
    geno_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can'
    
    # CSV file with SNP indices
    csv_file = '/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/extracted/brea/brea_0.1.csv'
    
    # First 5 columns file
    first_5_col_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen.gz'
    
    # New directories for updated files
    updated_geno_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_stats_info_0.1'
    
    # Read SNP indices from CSV
    snp_indices = read_snp_indices(csv_file)
    
    
    # Prepare arguments for multiprocessing
    args_list = []
    
    # Add .gen.gz files from col directory
    for filename in os.listdir(geno_dir):
        if filename.endswith('.gen.gz'):
            input_file = os.path.join(geno_dir, filename)
            output_file = os.path.join(updated_geno_dir, filename)
            args_list.append((input_file, output_file, snp_indices, 3))
    
   # Add first_5_col_col_gen.gz file
    input_file = first_5_col_file
    file_name, file_ext = os.path.splitext(first_5_col_file)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    output_file = f"{file_name}_stats_info_0.1.gen.gz"
    args_list.append((input_file, output_file, snp_indices, 5))
    
    # Use multiprocessing to process files
    with Pool(processes=cpu_count()-3) as pool:
        results = pool.map(process_gz_file, args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()