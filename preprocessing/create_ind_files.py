import os
import gzip
import argparse
import time

# Argument Parser
parser = argparse.ArgumentParser(description='Generating Genotype Files per Individual')
parser.add_argument('-in_dict', type=str, default='/vol/vssp/SF_ucdatasets/gwas', help='Path to the folder with .gen.gz files')
parser.add_argument('-out_dict', type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_ind', help='Path to folder storing per sample files')
parser.add_argument('-start_idx', type=int, default= 76655, help='Start index of the sample SNP data.')
parser.add_argument('-sample_idx', type=int, default=25551, help='Index of the sample.')


args = parser.parse_args()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  
    except ValueError:
        chr_num = float('inf')
    
    return (chr_num, file_name)

def process_genotype_file(input_folder, output_folder, start_index, sample_idx):
    start_time = time.time()

    files = os.listdir(input_folder)
    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)

    sample = sample_idx
    output_file_path = os.path.join(output_folder, f"sample_{sample}.gen.gz")
    num_snps = 0

    # Open the genotype file
    with gzip.open(output_file_path, 'wt') as out_handler:
        for gen_file in sorted_files:
            num_lines = 0
            print(f"Processing File:\n {gen_file}")
            with gzip.open(os.path.join(input_folder, gen_file), 'rt') as in_handler:
                #output_lines = set()
                for line in in_handler:
                    num_lines += 1
                    parts = line.split()
                    end_index = start_index + 3
                    selected_columns = parts[:5] + parts[start_index:end_index]
                    selected_line = ' '.join(selected_columns)
                    out_handler.write(selected_line + '\n')
            num_columns = len(parts)
            print("Shape of data in", gen_file, ":", num_lines, "rows x", num_columns, "columns.")
            print("Added", num_lines, "SNPs to the ", output_file_path, "file.")
            num_snps += num_lines
    print(f"Total number of SNPs merged is: \n {num_snps}")
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")

if __name__ == "__main__":
    input_folder = args.in_dict
    output_folder = args.out_dict
    start_index = args.start_idx
    sample_idx = args.sample_idx
    process_genotype_file(input_folder, output_folder, start_index, sample_idx)

# def process_genotype_file(file_path, output_folder, start_index):
#     # Open the genotype file
#     with gzip.open(file_path, 'rt') as f:
#         # Read all lines
#         lines = f.readlines()

#     # Process each line
#     output_lines = set()
#     for line in lines:
#         # Split the line into parts based on tab
#         parts = line.split()
#         end_index = args.start_idx+3
#         # Extract the first 5 columns and the selected range
#         selected_columns = parts[:5] + parts[start_index:end_index]
       
#         # Join the columns back into a string
#         selected_line = ' '.join(selected_columns)
#         print(f"Selected Line is: \n {selected_line}")
#         output_lines.add(selected_line)

#     print(f"Output Lines are: \n {output_lines}")
#     sample = args.sample_idx
#     output_file_path = os.path.join(output_folder, f"sample_{sample}.gen.gz")

#     with gzip.open(output_file_path, 'at') as f:
#         for line in output_lines:
#             f.write(line + '\n')


