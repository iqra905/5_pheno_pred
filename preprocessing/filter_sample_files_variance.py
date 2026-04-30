# # *********************** For All Split Folders ***********************
# import os
# import gzip
# import csv
# import pandas as pd
# import glob
# from multiprocessing import Pool, cpu_count

# def read_snp_indices(csv_file):
#     snp_indices = set()
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             snp_indices.add(int(row['SNP_index_all']))
#     return snp_indices

# def process_gz_file(args):
#     input_file, output_file, snp_indices, num_columns = args
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
#         for i, line in enumerate(in_f, 0):
#             if i in snp_indices:
#                 out_f.write(line)
#     return f"Processed {input_file} -> {output_file}"

# def main():
#     base_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea/stat_res_splits_random_reverse'
#     split_folders = glob.glob(os.path.join(base_path, 'split_*'))
    
#     for split_folder in split_folders:
#         split_name = os.path.basename(split_folder)
        
#         # Input paths
#         col_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea/geno_ml'
#         csv_file = os.path.join(split_folder, 'all_snps_chr_merged_0.05.csv')
#         first_5_col_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen.gz'
        
#         # Output paths
#         updated_col_dir = f'/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea/geno_ml_filtered_splits_random_reverse/{split_name}'
        
#         # Check if output directory exists
#         if not os.path.exists(updated_col_dir):
#             os.makedirs(updated_col_dir)
#             print(f"Created directory: {updated_col_dir}")
        
#         if not os.path.exists(csv_file):
#             print(f"Missing CSV file: {csv_file}")
#             continue
            
#         # Read SNP indices
#         snp_indices = read_snp_indices(csv_file)
        
#         # Prepare arguments for multiprocessing
#         args_list = []
        
#         # Add .gen.gz files
#         for filename in os.listdir(col_dir):
#             if filename.endswith('.gen.gz'):
#                 input_file = os.path.join(col_dir, filename)
#                 output_file = os.path.join(updated_col_dir, filename)
#                 args_list.append((input_file, output_file, snp_indices, 3))
        
#         # Add first_5_col file
#         output_file = f"/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea/geno_ml_filtered_splits_random_reverse/first_5_columns_brea_stat_exp_{split_name}_0.05.gen.gz"
#         args_list.append((first_5_col_file, output_file, snp_indices, 5))
        
#         # Process files
#         with Pool(processes=cpu_count()) as pool:
#             results = pool.map(process_gz_file, args_list)
        
#         for result in results:
#             print(result)
        
#         print(f"Completed processing {split_name}")

# if __name__ == '__main__':
#     main()

#********************* For a single Folder ******************
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

def main():
    # Directory containing .gen.gz files
    col_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can'
    
    # CSV file with SNP indices
    csv_file = '/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/snp_variance_results/brea_20/high_variance_snps_all_chromosomes.csv'
    
    # First 5 columns file
    first_5_col_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen.gz'
    
    # New directories for updated files
    updated_col_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_var_20'
    #updated_first_5_col_dir = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/'
    
    # Read SNP indices from CSV
    snp_indices = read_snp_indices(csv_file)
    
    
    # Prepare arguments for multiprocessing
    args_list = []
    
    # Add .gen.gz files from col directory
    for filename in os.listdir(col_dir):
        if filename.endswith('.gen.gz'):
            input_file = os.path.join(col_dir, filename)
            output_file = os.path.join(updated_col_dir, filename)
            args_list.append((input_file, output_file, snp_indices, 3))
    
   # Add first_5_col_col_gen.gz file
    input_file = first_5_col_file
    file_name, file_ext = os.path.splitext(first_5_col_file)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    output_file = f"{file_name}_variance_20.gen.gz"
    #output_file = f"/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea/geno_ml_filtered_splits/first_5_columns_brea_stat_exp_96_0.05.gen.gz"
    args_list.append((input_file, output_file, snp_indices, 5))
    
    # Use multiprocessing to process files
    with Pool(processes=cpu_count()-3) as pool:
        results = pool.map(process_gz_file, args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()