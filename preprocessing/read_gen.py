#################### Final Version with multiprocessing to get the index of incorrect columns #########################

import os
import gzip
import time
import multiprocessing as mp

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  
    except ValueError:
        chr_num = float('inf')
    
    return (chr_num, file_name)

def count_snps_in_file(file_path):
    snp_count = 0
    incorrect_lines = []
    with gzip.open(file_path, 'rt') as f:
        for index, line in enumerate(f):
            columns = len(line.split())
            if columns != 8:
                incorrect_lines.append((index, columns))
            snp_count += 1
    return file_path, snp_count, incorrect_lines

def process_file(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    return count_snps_in_file(file_path)

def count_total_snps(folder_path, output_file_path):
    start_time = time.time()

    files = os.listdir(folder_path)
    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)
    print(f"Found {len(sorted_files)} genotype files.")

    with open(output_file_path, 'w') as f:
        f.write("File Name, Number of SNPs, Incorrect Lines (Index: Column Count)\n")

    # Create a pool of workers
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Map the process_file function to all files
        results = pool.starmap(process_file, [(file_name, folder_path) for file_name in sorted_files])

    # Process results
    for file_path, snp_count, incorrect_lines in results:
        file_name = os.path.basename(file_path)
        print(f"Processed File: {file_name}")

        incorrect_lines_str = ', '.join(f"{index}:{columns}" for index, columns in incorrect_lines)
        
        with open(output_file_path, 'a') as f:
            f.write(f"{file_name}, {snp_count}: \n {incorrect_lines_str}\n")
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")

if __name__ == "__main__":
    folder_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_SNPs_unq'
    output_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_SNPs_unq/sample_files_5M_SNPs_unq.txt'
    count_total_snps(folder_path, output_file_path)

# import os
# import gzip
# import time
# import multiprocessing as mp

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def count_snps_in_file(file_path):
#     snp_count = 0
#     column_counts = []
#     with gzip.open(file_path, 'rt') as f:
#         for line in f:
#             columns = len(line.split())
#             column_counts.append(columns)
#             snp_count += 1
#     return file_path, snp_count, column_counts

# def process_file(file_name, folder_path, output_file_path):
#     file_path = os.path.join(folder_path, file_name)
#     result = count_snps_in_file(file_path)
    
#     # Print and write processing info
#     file_path, snp_count, column_counts = result
#     print(f"Processed File:\n {file_name}")
#     column_info = ', '.join(map(str, column_counts))
    
#     with open(output_file_path, 'a') as f:
#         f.write(f"{file_name}: {snp_count} SNPs with columns: {column_info}\n")
    
#     return result

# def count_total_snps(folder_path, output_file_path):
#     start_time = time.time()

#     files = os.listdir(folder_path)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
#     print(f"Found {len(sorted_files)} genotype files.")

#     with open(output_file_path, 'w') as f:
#         f.write("Number of SNPs and columns per file:\n")

#     # Create a pool of workers
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         # Map the process_file function to all files
#         results = pool.starmap(process_file, [(file_name, folder_path, output_file_path) for file_name in sorted_files])

#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print("Execution time:", elapsed_time, "seconds")

# if __name__ == "__main__":
#     folder_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data'
#     output_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sample_files_info.txt'
#     count_total_snps(folder_path, output_file_path)
    
################################## Reading files for rows and columns without multiprocessing ###########################
# import os
# import gzip
# import time

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def count_snps_in_file(file_path):
#     snp_count = 0
#     column_counts = []
#     with gzip.open(file_path, 'rt') as f:
#         for line in f:
#             columns = len(line.split())
#             column_counts.append(columns)
#             snp_count += 1
#     return snp_count, column_counts

# def count_total_snps(folder_path, output_file_path):
#     start_time = time.time()
#     total_snps = 0

#     files = os.listdir(folder_path)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
#     print(f"Found {len(sorted_files)} genotype files.")

#     with open(output_file_path, 'w') as f:
#         f.write("Number of SNPs and columns per file:\n")

#     for file_name in sorted_files:
#         print(f"Processing File:\n {file_name}")
#         file_path = os.path.join(folder_path, file_name)
#         snp_count, column_counts = count_snps_in_file(file_path)

#         # total_snps += snp_count

#         column_info = ', '.join(map(str, column_counts))
        
#         with open(output_file_path, 'a') as f:
#             f.write(f"{file_name}: {snp_count} SNPs with columns: {column_info}\n")
    
#     # with open(output_file_path, 'a') as f:
#     #     f.write(f"\nTotal number of SNPs in all .gen.gz files: {total_snps}\n")
    
#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print("Execution time:", elapsed_time, "seconds")

# if __name__ == "__main__":
#    folder_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data'
#    output_file_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sample_files_info.txt'
#    count_total_snps(folder_path, output_file_path)



##################### Final Version without multiprocessing to get the index of incorrect columns #########################
# import os
# import gzip
# import time

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def count_snps_in_file(file_path):
#     snp_count = 0
#     incorrect_lines = []
#     with gzip.open(file_path, 'rt') as f:
#         for index, line in enumerate(f):
#             columns = len(line.split())
#             #if columns != 112994:
#             if columns != 8:
#                 incorrect_lines.append((index, columns))
#             snp_count += 1
#     return snp_count, incorrect_lines

# def count_total_snps(folder_path, output_file_path):
#     start_time = time.time()
#     total_snps = 0

#     files = os.listdir(folder_path)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
#     print(f"Found {len(sorted_files)} genotype files.")

#     with open(output_file_path, 'w') as f:
#         f.write("File Name, Number of SNPs, Incorrect Lines (Index: Column Count)\n")

#     for file_name in sorted_files:
#         print(f"Processing File: {file_name}")
#         file_path = os.path.join(folder_path, file_name)
#         snp_count, incorrect_lines = count_snps_in_file(file_path)

#         #total_snps += snp_count

#         incorrect_lines_str = ', '.join(f"{index}:{columns}" for index, columns in incorrect_lines)
        
#         with open(output_file_path, 'a') as f:
#             f.write(f"{file_name}, {snp_count}: \n {incorrect_lines_str}\n")
    
#     #with open(output_file_path, 'a') as f:
#         #f.write(f"\nTotal number of SNPs in all .gen.gz files: {total_snps}\n")
    
#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print("Execution time:", elapsed_time, "seconds")

# if __name__ == "__main__":
#     folder_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data'
#     output_file_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sample_files_info.txt'
#     # folder_path = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen'
#     # output_file_path = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen/Data_Files_Per_SNPs_Info.txt'
#     count_total_snps(folder_path, output_file_path)



# import os
# import gzip
# import time

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def count_snps_in_file(file_path):
#     snp_count = 0
#     first_line_columns = None
#     last_line_columns = None
#     with gzip.open(file_path, 'rt') as f:
#         for line in f:
#             if snp_count == 0:
#                 first_line_columns = len(line.split())
#             snp_count += 1
#         last_line_columns = len(line.split())
#     return snp_count, first_line_columns, last_line_columns


# def count_total_snps(folder_path, output_file_path):
#     start_time = time.time()
#     total_snps = 0

#     files = os.listdir(folder_path)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
#     print(f"Found {len(sorted_files)} genotype files.")

#     with open(output_file_path, 'w') as f:
#         f.write("Number of SNPs and columns per file:\n")

#     for file_name in sorted_files:
#         print(f"Processing File:\n {file_name}")
#         file_path = os.path.join(folder_path, file_name)
#         snp_count, first_line_columns, last_line_columns = count_snps_in_file(file_path)

#         total_snps += snp_count

#         col_count = first_line_columns if first_line_columns == last_line_columns else f"{first_line_columns} -> {last_line_columns}"
                
        
#         with open(output_file_path, 'a') as f:
#             if first_line_columns != last_line_columns:
#                 f.write(f"{file_name}: {snp_count} SNPs X {col_count} columns\n")
#             else:
#                 f.write(f"{file_name}: {snp_count} SNPs X {col_count} columns\n")
    
#     with open(output_file_path, 'a') as f:
#         f.write(f"\nTotal number of SNPs in all .gen.gz files: {total_snps}\n")
    
#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print("Execution time:", elapsed_time, "seconds")
    

# if __name__ == "__main__":
#     #folder_path1 = '/vol/research/ucdatasets/gwas/error_chunks'
#     folder_path = '/vol/research/ucdatasets/gwas/error_chunks'
#     output_file_path = '/vol/research/ucdatasets/gwas/Data-Files_SNPs_Info.txt'
#     count_total_snps(folder_path, output_file_path)


