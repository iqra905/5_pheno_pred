# # #***************************** Process sample files for all diseases and thresholds **********************
# import os
# import gzip
# import csv
# import pandas as pd
# import argparse
# from multiprocessing import Pool, cpu_count

# def read_snp_indices(csv_file):
#     snp_indices = set()
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             snp_indices.add(int(row['Global_SNP_Index']))
#     return snp_indices

# def process_gz_file(args):
#     input_file, output_file, snp_indices, num_columns = args
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with gzip.open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
#         for i, line in enumerate(in_f, 0):
#             if i in snp_indices:
#                 out_f.write(line)
#     return f"Processed {input_file} -> {output_file}"

# def process_first_5_cols(args):
#    input_file, output_file, global_snp_indices = args
    
#    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#    with open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
#       for i, line in enumerate(in_f, 0):  # 0-based indexing
#         if i in global_snp_indices:
#             out_f.write(line)
    
#    return f"Processed first 5 columns file: {input_file} -> {output_file}"

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Filter genomic data based on SNP indices.')
    
#     # Required arguments
#     parser.add_argument('-geno_dir', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/t2d', help='Directory containing .gen.gz files')
#     parser.add_argument('-first_5_col_file', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen',help='Path to the first 5 columns file')
#     parser.add_argument('-disease', default='t2d', help='Disease name (e.g., t2d)')
#     parser.add_argument('-threshold', default =0.01,help='Threshold value (e.g., 0.01)')
    
#     return parser.parse_args()

# def main():
#     args = parse_arguments()

#     geno_dir = args.geno_dir
#     first_5_col_file = args.first_5_col_file
#     disease = args.disease
#     threshold = args.threshold
    
#     # CSV file with SNP indices
#     csv_file = f'/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/gen_data_5M_filtered_plink_files/{disease}/{disease}_{threshold}_LDpruned.csv'
        
#     # New directories for updated files
#     updated_geno_dir = f'/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/data_new_study_split_20/sampled_data_5M_disease_wise_pruned/{disease}/{threshold}'

#     print(f"Input directory: {geno_dir}")
#     print(f"Output directory: {updated_geno_dir}")
#     print(f"CSV file: {csv_file}")
#     print(f"Disease: {disease}")
#     print(f"Threshold: {threshold}")
    
#     # Read SNP indices from CSV
#     snp_indices = read_snp_indices(csv_file)

#     # Process first 5 columns file
#     input_file = first_5_col_file
    
#     # file_name, file_ext = os.path.splitext(first_5_col_file)
#     # if file_ext == '.gz':
#     #     file_name, _ = os.path.splitext(file_name)  
#     # output_file = f"{file_name}_pruned_split_{disease}_{threshold}.gen.gz"
    
#     # Extract just the base filename without path or extension
#     base_filename = os.path.basename(first_5_col_file)
#     file_name, file_ext = os.path.splitext(base_filename)
#     if file_ext == '.gz':
#         file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    
#     # Save the output file in the updated_geno_dir
#     output_file = os.path.join(updated_geno_dir, f"first_5_columns_pruned_split_20_{disease}_{threshold}.gen.gz")
    

#     # Process the first 5 columns file separately
#     result = process_first_5_cols((input_file, output_file, snp_indices))
#     print(result)
    
    
#     # Prepare arguments for multiprocessing
#     args_list = []
    
#     # Add .gen.gz files from geno directory
#     for filename in os.listdir(geno_dir):
#         if filename.endswith('.gen.gz'):
#             input_file = os.path.join(geno_dir, filename)
#             output_file = os.path.join(updated_geno_dir, filename)
#             args_list.append((input_file, output_file, snp_indices, 3))
        
#     # Use multiprocessing to process files
#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(process_gz_file, args_list)
    
#     # Print results
#     for result in results:
#         print(result)

# if __name__ == '__main__':
#     main()

###################################################################################################################################

#***************************** Process sample files for a single folder **********************
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

def process_first_5_cols(args):
   input_file, output_file, global_snp_indices = args
    
   os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
   with open(input_file, 'rt') as in_f, gzip.open(output_file, 'wt') as out_f:
      for i, line in enumerate(in_f, 0):  # 0-based indexing
         if i in global_snp_indices:
            out_f.write(line)
    
   return f"Processed first 5 columns file: {input_file} -> {output_file}"

def main():
    # Directory containing .gen.gz files
    geno_dir = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/t2d'

    # CSV file with SNP indices
    #csv_file = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M_filtered_plink_files/t2d/t2d_0.01_LDpruned.csv'
    csv_file = '/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/snp_overlap_analysis_csv/ldpruned_t2d_pval0.1_unique_snps_full_data.csv'
    
    # First 5 columns file
    first_5_col_file = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
    
    # New directories for updated files
    #updated_geno_dir = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned/t2d_0.01'
    updated_geno_dir = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis/ldpruned_t2d_pval0.1_unique_snps_full_data'
    
    # Read SNP indices from CSV
    snp_indices = read_snp_indices(csv_file)

   # Process first 5 columns file
    input_file = first_5_col_file

    # file_name, file_ext = os.path.splitext(first_5_col_file)
    # if file_ext == '.gz':
    #     file_name, _ = os.path.splitext(file_name)  
    # #output_file = f"{file_name}_pruned_t2d_0.01.gen.gz"
    # output_file = f"{file_name}_ldpruned_t2d_pval0.1_unique_snps_full_data.gen.gz"

    # Extract just the base filename without path or extension
    base_filename = os.path.basename(first_5_col_file)
    file_name, file_ext = os.path.splitext(base_filename)
    if file_ext == '.gz':
        file_name, _ = os.path.splitext(file_name)  # Remove .gz extension
    
    # Save the output file in the updated_geno_dir
    output_file = os.path.join(updated_geno_dir, f"first_5_columns_5M_updated_unq_ldpruned_t2d_pval0.1_unique_snps_full_data.gen.gz")
    
    # Process the first 5 columns file separately
    result = process_first_5_cols((input_file, output_file, snp_indices))
    print(result)
    
    
    # Prepare arguments for multiprocessing
    args_list = []
    
    # Add .gen.gz files from geno directory
    for filename in os.listdir(geno_dir):
        if filename.endswith('.gen.gz'):
            input_file = os.path.join(geno_dir, filename)
            output_file = os.path.join(updated_geno_dir, filename)
            args_list.append((input_file, output_file, snp_indices, 3))
        
    # Use multiprocessing to process files
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_gz_file, args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
