#***************************************** Process all disease folders in loop ***********************************#
# import os
# import gzip
# import csv
# import pandas as pd
# from multiprocessing import Pool, cpu_count

# def read_snp_indices_by_chr(csv_file):
#    # Dictionary to store SNP indices for each chromosome
#    chr_snp_indices = {str(int(i)): set() for i in range(1, 23)}
   
#    with open(csv_file, 'r') as f:
#        reader = csv.DictReader(f)
#        for row in reader:
#            chr_num = str(int(float(row['Chromosome'])))
#            snp_idx = int(row['SNP_Index'])
#            chr_snp_indices[chr_num].add(snp_idx)
#    return chr_snp_indices

# def process_gz_file(args):
#    input_file, output_file, chr_snp_indices, chr_num = args

#    # Get the specific indices for this chromosome
#    snp_indices = chr_snp_indices[chr_num]
   
#    os.makedirs(os.path.dirname(output_file), exist_ok=True)

#    with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
#        for i, line in enumerate(in_f, 0):
#            if i in snp_indices:
#                out_f.write(line)
   
#    return f"Processed chromosome {chr_num}: {input_file} -> {output_file}"

# def process_first_5_cols(args):
#    input_file, output_file, chr_snp_indices = args
   
#    os.makedirs(os.path.dirname(output_file), exist_ok=True)
   
#    # Combine all SNP indices for matching against the first 5 columns file
#    all_indices = set()
#    for indices in chr_snp_indices.values():
#        all_indices.update(indices)
   
#    with open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
#         for i, line in enumerate(in_f, 0):
#             if i in all_indices:
#                 out_f.write(line)

# #    with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
# #        for i, line in enumerate(in_f, 0):
# #            if i in all_indices:
# #                out_f.write(line)
   
#    return f"Processed first 5 columns file: {input_file} -> {output_file}"

# def process_disease(disease):
#    # Base directories
#    base_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm'
#    results_dir = '/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/_split'
   
#    # Disease specific paths
#    col_dir = f'{base_dir}/gen_data_5M/merged/chr_wise'
#    csv_file = f'{results_dir}/{disease}/all_snps_chr_merged_0.01.csv'
#    first_5_col_file = f'{base_dir}/meta_data/first_5_columns_5M_updated_unq.gen'
#    updated_col_dir = f'{base_dir}/gen_data_5M_filtered/{disease}/0.01'
   
#    # Read SNP indices
#    chr_snp_indices = read_snp_indices_by_chr(csv_file)
   
#    # Process chromosome files
#    args_list = []
#    for filename in os.listdir(col_dir):
#        if filename.endswith('.gen.gz'):
#           # Extract chromosome number from filename
#            chr_num = filename.split('chr')[1].split('.')[0]
#            if chr_num in chr_snp_indices:
#                input_file = os.path.join(col_dir, filename)
#                output_file = os.path.join(updated_col_dir, filename)
#                args_list.append((input_file, output_file, chr_snp_indices, chr_num))
   
#    # Process chromosome files
#    with Pool(processes=cpu_count()) as pool:
#        results = pool.map(process_gz_file, args_list)
   
#    # Print chromosome processing results
#    for result in results:
#        print(result)
   
#    # Process first 5 columns file
#    file_name, file_ext = os.path.splitext(first_5_col_file)
#    if file_ext == '.gz':
#        file_name, _ = os.path.splitext(file_name)
#    output_file = f"{file_name}_gen_{disease}_0.01.gen.gz"
   
#    # Process the first 5 columns file separately
#    result = process_first_5_cols((first_5_col_file, output_file, chr_snp_indices))
#    print(result)

# def main():
#    diseases = ['pros', 'pan', 'col', 'brea', 't2d']
#    for disease in diseases:
#        print(f"\nProcessing {disease}...")
#        process_disease(disease)

# if __name__ == '__main__':
#    main()
   
#***************************************** Process single disease folder ***********************************#
import os
import gzip
import csv
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count

def read_snp_indices(csv_file):
   # Dictionary to store SNP indices for each chromosome
   chr_snp_indices = {str(int(i)): set() for i in range(1, 23)}
   
   # Set to store global SNP indices
   global_snp_indices = set()
   
   with open(csv_file, 'r') as f:
      reader = csv.DictReader(f)
      for row in reader:
         # Get chromosome-specific SNP index
         chr_num = str(int(float(row['Chromosome'])))
         snp_idx = int(row['SNP_Index'])
         chr_snp_indices[chr_num].add(snp_idx)
         
         # Get global SNP index if available
         if 'SNP_index_all' in row:
               global_idx = int(row['SNP_index_all'])
               global_snp_indices.add(global_idx)
   
   return chr_snp_indices, global_snp_indices
    

def process_gz_file(args):
    input_file, output_file, chr_snp_indices, chr_num = args
    
    # Get the specific indices for this chromosome
    snp_indices = chr_snp_indices[chr_num]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
        for i, line in enumerate(in_f, 0):  # 0-based indexing
            if i in snp_indices:
                out_f.write(line)
    
    return f"Processed chromosome {chr_num}: {input_file} -> {output_file}"

def process_first_5_cols(args):
   input_file, output_file, global_snp_indices = args
    
   os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
   with open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
      for i, line in enumerate(in_f, 0):  # 0-based indexing
         if i in global_snp_indices:
            out_f.write(line)
    
   return f"Processed first 5 columns file: {input_file} -> {output_file}"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Filter genomic data based on SNP indices.')
    
    # Required arguments
    parser.add_argument('-geno_dir', default='/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise', 
                        help='Directory containing .gen.gz files')
    parser.add_argument('-first_5_col_file', default='/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen',
                        help='Path to the first 5 columns file')
    parser.add_argument('-disease', default='t2d', help='Disease name (e.g., pros)')
    parser.add_argument('-threshold', default =0.1,
                        help='Threshold value (e.g., 0.01)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up variables based on arguments
    geno_dir = args.geno_dir
    disease = args.disease
    threshold = args.threshold
    first_5_col_file = args.first_5_col_file
    
    # Construct CSV file path
    csv_file = f'/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new_study_split_20/{disease}/all_snps_chr_merged_{threshold}.csv'
    
    # Construct updated geno directory
    updated_geno_dir = f'/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered/{disease}/{threshold}'
    
    print(f"Input directory: {geno_dir}")
    print(f"Output directory: {updated_geno_dir}")
    print(f"CSV file: {csv_file}")
    print(f"Disease: {disease}")
    print(f"Threshold: {threshold}")
    
    # Read SNP indices from CSV, organized by chromosome
    chr_snp_indices, global_snp_indices = read_snp_indices(csv_file)

    # # Process first 5 columns file
    # input_file = first_5_col_file
    # file_name, file_ext = os.path.splitext(first_5_col_file)
    # if file_ext == '.gz':
    #     file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    # output_file = f"{file_name}_gen_split_20_{disease}_{threshold}.gen.gz"

    # Process first 5 columns file
    input_file = first_5_col_file
    
    # Extract just the base filename without path or extension
    base_filename = os.path.basename(first_5_col_file)
    file_name, file_ext = os.path.splitext(base_filename)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    
    # Save the output file in the updated_geno_dir
    output_file = os.path.join(updated_geno_dir, f"first_5_columns_gen_split_20_{disease}_{threshold}.gen.gz")
    
    # Process the first 5 columns file separately
    result = process_first_5_cols((input_file, output_file, global_snp_indices))
    print(result)

    
    # Prepare arguments for multiprocessing
    args_list = []
    
    # Add .gen.gz files from geno directory
    for filename in os.listdir(geno_dir):
        if filename.endswith('.gen.gz'):
            # Extract chromosome number from filename
            chr_num = filename.split('chr')[1].split('.')[0]
            if chr_num in chr_snp_indices:
                input_file = os.path.join(geno_dir, filename)
                output_file = os.path.join(updated_geno_dir, filename)
                args_list.append((input_file, output_file, chr_snp_indices, chr_num))
    
    # Process chromosome files
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_gz_file, args_list)
    
    # Print chromosome processing results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
