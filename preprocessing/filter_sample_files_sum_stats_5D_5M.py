import os
import gzip
import csv
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count

def read_snp_indices(csv_file):
    snp_indices = set()
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snp_indices.add(int(row['Global_SNP_Index']))
    return snp_indices

def process_gz_file(args):
    input_file, output_file, snp_indices, num_columns = args
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
        for i, line in enumerate(in_f, 0):
            if i in snp_indices:
                out_f.write(line)
    return f"Processed {input_file} -> {output_file}"

def process_first_5_cols(args):
   input_file, output_file, global_snp_indices = args
    
   os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
   with open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
      for i, line in enumerate(in_f, 0):  # 0-based indexing
         if i in global_snp_indices:
            out_f.write(line)
    
   return f"Processed first 5 columns file: {input_file} -> {output_file}"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Filter genomic data files based on SNP indices.')
    parser.add_argument('-geno_dir', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros',  help='Directory containing .gen.gz files')
    parser.add_argument('-csv_file', default='/vol/vssp/SF_ucdatasets/gwas/data_files/5D_snp_info_files/extracted_5M/pros_0.01.csv', help='CSV file with SNP indices')
    parser.add_argument('-first_5_col_file', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen', help='First 5 columns file')
    parser.add_argument('-output_dir', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise_summ_stats/pros/0.01', help='Output directory for updated files')
    parser.add_argument('-first_5_col_output_dir', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise_summ_stats',help='Output directory for the first 5 columns file')
    
    parser.add_argument('-disease', default='pros', help='Disease name (e.g., pros)')
    parser.add_argument('-pval', default='0.01', help='P-value threshold (default: 0.01)')
    
    parser.add_argument('-num_columns', type=int, default=3, help='Number of columns to process (default: 3)')
    
    parser.add_argument('-processes', type=int, default=cpu_count(), help=f'Number of processes to use (default: {cpu_count()})')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.first_5_col_output_dir, exist_ok=True)
    
    # Read SNP indices from CSV
    snp_indices = read_snp_indices(args.csv_file)
   
    # Process first 5 columns file
    input_file = args.first_5_col_file

    # Extract just the base filename without path or extension
    base_filename = os.path.basename(args.first_5_col_file)
    file_name, file_ext = os.path.splitext(base_filename)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    
    # Save the output file for first 5 columns in a separate directory
    output_file = os.path.join(
        args.first_5_col_output_dir,
        f"first_5_columns_5M_updated_unq_{args.disease}_summ_stats_pval_{args.pval}.gen.gz"
    )
    
    # Process the first 5 columns file separately
    result = process_first_5_cols((input_file, output_file, snp_indices))
    print(result)
    
    # Prepare arguments for multiprocessing
    process_args_list = []
    
    # Add .gen.gz files from geno directory
    for filename in os.listdir(args.geno_dir):
        if filename.endswith('.gen.gz'):
            input_file = os.path.join(args.geno_dir, filename)
            output_file = os.path.join(args.output_dir, filename)
            process_args_list.append((input_file, output_file, snp_indices, args.num_columns))
    
    # Use multiprocessing to process files
    with Pool(processes=args.processes) as pool:
        results = pool.map(process_gz_file, process_args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()