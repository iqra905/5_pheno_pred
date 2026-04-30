import os
import gzip
import time
import multiprocessing
from functools import partial

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  
    except ValueError:
        chr_num = float('inf')
    
    return (chr_num, file_name)
def process_and_count_snps(file_path, output_path, condition):
    snp_count = 0
    skipped_lines = []
    current_index = 0
    previous_line = None

    try:
        with gzip.open(file_path, 'rt') as f_in, gzip.open(output_path, 'wt') as f_out:
            for line in f_in:
                if condition == 'remove_last_line':
                    if previous_line is not None:
                        f_out.write(previous_line)
                        snp_count += 1
                    previous_line = line
                elif condition == 'remove_index_47495' and current_index == 47495:
                    skipped_lines.append(current_index)
                    print(f"Skipping line {current_index} in file {file_path}")
                elif condition == 'remove_index_38846' and current_index == 38846:
                    skipped_lines.append(current_index)
                    print(f"Skipping line {current_index} in file {file_path}")
                elif condition == 'remove_index_44943' and current_index == 44943:
                    skipped_lines.append(current_index)
                    print(f"Skipping line {current_index} in file {file_path}")
                elif condition == 'remove_index_40062' and current_index == 40062:
                    skipped_lines.append(current_index)
                    print(f"Skipping line {current_index} in file {file_path}")
                else:
                    f_out.write(line)
                    snp_count += 1
                
                current_index += 1

            if condition == 'remove_last_line':
                skipped_lines.append(current_index - 1)
                print(f"Skipped last line (index {current_index - 1}) in file {file_path}")

        return file_path, snp_count, skipped_lines

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, 0, skipped_lines
    
# def process_and_count_snps(file_path, output_path, condition):
#     snp_count = 0
#     skipped_lines = []

#     try:
#         with gzip.open(file_path, 'rt') as f_in, gzip.open(output_path, 'wt') as f_out:
#             for index, line in enumerate(f_in):
#                 if condition == 'remove_last_line' and index == -1:
#                     skipped_lines.append(index)
#                     print(f"Skipping {index} line in file {file_path}")
#                     continue
#                 elif condition == 'remove_index_47495' and index == 47495:
#                     skipped_lines.append(index)
#                     print(f"Skipping {index} in file {file_path}")
#                     continue
#                 elif condition == 'remove_index_38846' and index == 38846:
#                     skipped_lines.append(index)
#                     print(f"Skipping {index} in file {file_path}")
#                     continue
#                 elif condition == 'remove_index_44943' and index == 44943:
#                     skipped_lines.append(index)
#                     print(f"Skipping {index} in file {file_path}")
#                     continue
#                 elif condition == 'remove_index_40062' and index == 40062:
#                     skipped_lines.append(index)
#                     print(f"Skipping {index} in file {file_path}")
#                     continue
#                 else:
#                     f_out.write(line)
#                     snp_count += 1

#         return file_path, snp_count, skipped_lines

#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#         return file_path, 0, skipped_lines

def process_file(file_name, folder_path, output_folder_path, output_file_path):
    print(f"Processing File: {file_name}")
    file_path = os.path.join(folder_path, file_name)
    output_path = os.path.join(output_folder_path, file_name)
    
    file_base_name = file_name.split('.')[0]
    if file_base_name.endswith('e'):
        condition = 'remove_last_line'
    elif file_base_name == 'chr1_4a':
        condition = 'remove_index_47495'
    elif file_base_name == 'chr13_3a':
        condition = 'remove_index_38846'
    elif file_base_name == 'chr17_1a':
        condition = 'remove_index_44943'
    elif file_base_name == 'chr19_1a':
        condition = 'remove_index_40062'
    else:
        condition = None
    
    file_path, snp_count, skipped_lines = process_and_count_snps(file_path, output_path, condition)
    
    with open(output_file_path, 'a') as f:
        if skipped_lines:
            skip_info = f"{file_name}: {snp_count}: Skipped Indexes: {skipped_lines}\n"
            f.write(skip_info)
            print(f"{file_name}: {snp_count}: Skipped Indexes: {skipped_lines}\n")
        else:
            count_info = f"{file_name}: {snp_count}\n"
            f.write(count_info)
            print(f"{file_name}: {snp_count}")
    
    return file_name, snp_count

def process_files(folder_path, output_folder_path, output_file_path):
    start_time = time.time()

    try:
        files = os.listdir(folder_path)
        gen_files = [f for f in files if f.endswith('.gen.gz')]
        sorted_files = sorted(gen_files, key=custom_sort)
        print(f"Found {len(sorted_files)} genotype files.")

        with open(output_file_path, 'w') as f:
            f.write("File Name: SNP Count: Skipped Indexes (if any)\n")

        # Create a partial function with fixed arguments for the process_file function
        process_file_partial = partial(process_file, folder_path=folder_path, output_folder_path=output_folder_path, output_file_path=output_file_path)

        with multiprocessing.Pool() as pool:
            pool.map(process_file_partial, sorted_files)

    except Exception as e:
        print(f"Error processing files: {e}")

    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

if __name__ == "__main__":
    # folder_path = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen'
    # output_folder_path = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen/processed_chunks'
    # output_file_path = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen/Error_SNPs_Info_rm.txt'

    folder_path = '/vol/vssp/SF_ucdatasets/gwas/error_chunks'
    output_folder_path = '/vol/vssp/SF_ucdatasets/gwas/error_chunks/processed_chunks'
    output_file_path = '/vol/vssp/SF_ucdatasets/gwas/Error_SNPs_Info_rm.txt'

    os.makedirs(output_folder_path, exist_ok=True)

    process_files(folder_path, output_folder_path, output_file_path)
