#******************* PROCESSING ALL FILES IN A FOLDER MULTIPROCESSING - version 2 **************************#
import gzip
import os
import time
import argparse
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description='Remove duplicate SNP occurrences from sample files.')
parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind', help='Path to the folder with sample_n.gen.gz files')
parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')
parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
parser.add_argument('-slice', type=str, default='0:10000', help='Slice of files to process (e.g., "0:10000", "10000:20000", "20000:")')
args = parser.parse_args()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        sample_num = int(parts[1].split('.')[0])  
    except ValueError:
        sample_num = float('inf')
    
    return (sample_num, file_name)

def find_duplicates(input_file, unique_folder, duplicate_folder):
    try:
        start_time = time.time()
        unique_values = set()
        value_counts = {}
        num_lines = 0
        
        base_name = os.path.basename(input_file).rsplit('.', 2)[0]
        unique_file_path = os.path.join(unique_folder, f"{base_name}.gen.gz")
        duplicate_file_path = os.path.join(duplicate_folder, f"{base_name}.gen.gz")

        with gzip.open(input_file, 'rt') as file:
            with gzip.open(duplicate_file_path, 'wt') as duplicate_handle:
                with gzip.open(unique_file_path, 'wt') as unique_handle:
                    for line in file:
                        num_lines += 1
                        columns = line.split()
                        key = (columns[0], columns[2])  # Use first and third columns as key
                        value_counts[key] = value_counts.get(key, 0) + 1
                        if value_counts[key] == 1:
                            unique_values.add(key)
                            unique_handle.write(line)
                        else:
                            duplicate_handle.write(line)

        num_columns = len(columns)
        print(f"Shape of data in {input_file}: {num_lines} SNPs x {num_columns} columns")
        print(f"Based on cols Chr and Location:- Unique SNPs: {len(unique_values)}, Duplicate SNPs: {num_lines - len(unique_values)}\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Execution time for {input_file}: {elapsed_time} seconds")

    except FileNotFoundError:
        print(f"File not found: {input_file}. Please provide a valid file path.")

def process_file(file_name):
    input_folder = args.in_dict
    unique_folder = args.out_dict_unq
    duplicate_folder = args.out_dict_dup
    input_file_path = os.path.join(input_folder, file_name)
   # print(f"Processing File:\n {file_name}")
    find_duplicates(input_file_path, unique_folder, duplicate_folder)

def process_folder(input_folder, unique_folder, duplicate_folder, slice_str):
    os.makedirs(unique_folder, exist_ok=True)
    os.makedirs(duplicate_folder, exist_ok=True)

    files = os.listdir(input_folder)
    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)

    # Parse the slice string
    slice_parts = slice_str.split(':')
    start = int(slice_parts[0]) if slice_parts[0] else None
    end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None

    # Select subset of files based on the slice
    subset_files = sorted_files[start:end]

    print(f"Processing files {start or 0} to {end or 'end'}")
    print(f"Number of files to process: {len(subset_files)}")

    # Use multiprocessing to process files in parallel
    with Pool(cpu_count()) as pool:
        pool.map(process_file, subset_files)

if __name__ == "__main__":
    input_folder = args.in_dict
    unique_folder = args.out_dict_unq
    duplicate_folder = args.out_dict_dup
    slice_str = args.slice
    process_folder(input_folder, unique_folder, duplicate_folder, slice_str)

#********************************** PROCESSING ALL FILES IN A FOLDER ***********************************#
# import gzip
# import os
# import time
# import argparse
# from multiprocessing import Pool, cpu_count

# parser = argparse.ArgumentParser(description='Remove duplicate SNP occurrences from sample files.')
# parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind', help='Path to the folder with sample_n.gen.gz files')
# parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
# parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')
# args = parser.parse_args()

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         sample_num = int(parts[1].split('.')[0])  
#     except ValueError:
#         sample_num = float('inf')
    
#     return (sample_num, file_name)

# def find_duplicates(input_file, unique_folder, duplicate_folder):
#     try:
#         start_time = time.time()
#         unique_values = set()
#         value_counts = {}
#         num_lines = 0
        
#         base_name = os.path.basename(input_file).rsplit('.', 2)[0]
#         unique_file_path = os.path.join(unique_folder, f"{base_name}.gen.gz")
#         duplicate_file_path = os.path.join(duplicate_folder, f"{base_name}.gen.gz")

#         with gzip.open(input_file, 'rt') as file:
#             with gzip.open(duplicate_file_path, 'wt') as duplicate_handle:
#                 with gzip.open(unique_file_path, 'wt') as unique_handle:
#                     for line in file:
#                         num_lines += 1
#                         columns = line.split()
#                         column_3_value = columns[2]
#                         value_counts[column_3_value] = value_counts.get(column_3_value, 0) + 1
#                         if value_counts[column_3_value] == 1:
#                             unique_values.add(column_3_value)
#                             unique_handle.write(line)
#                         else:
#                             duplicate_handle.write(line)

#         num_columns = len(columns)
#         print(f"Shape of data in {input_file}: {num_lines} lines x {num_columns} columns")
#         print(f"Number of Unique lines based on column 3: {len(unique_values)}")
#         print(f"Number of duplicate lines based on column 3: {num_lines - len(unique_values)}")

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Execution time for {input_file}: {elapsed_time} seconds")

#     except FileNotFoundError:
#         print(f"File not found: {input_file}. Please provide a valid file path.")

# def process_file(file_name):
#     input_folder = args.in_dict
#     unique_folder = args.out_dict_unq
#     duplicate_folder = args.out_dict_dup
#     input_file_path = os.path.join(input_folder, file_name)
#     print(f"Processing File:\n {file_name}")
#     find_duplicates(input_file_path, unique_folder, duplicate_folder)

# def process_folder(input_folder, unique_folder, duplicate_folder):
#     os.makedirs(unique_folder, exist_ok=True)
#     os.makedirs(duplicate_folder, exist_ok=True)

#     files = os.listdir(input_folder)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)

#     # Use multiprocessing to process files in parallel
#     with Pool(cpu_count()) as pool:
#         pool.map(process_file, sorted_files)

# if __name__ == "__main__":
#     input_folder = args.in_dict
#     unique_folder = args.out_dict_unq
#     duplicate_folder = args.out_dict_dup
#     process_folder(input_folder, unique_folder, duplicate_folder)


#********************************** PROCESSING ALL FILES IN A FOLDER ***********************************#
# import gzip
# import os
# import time
# import argparse

# parser = argparse.ArgumentParser(description='Remove duplicate SNP ocurrences from sample files.')
# parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind', help='Path to the folder with sample_n.gen.gz files')
# parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
# parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')
# args = parser.parse_args()

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         sample_num = int(parts[1].split('.')[0])  
#     except ValueError:
#         sample_num = float('inf')
    
#     return (sample_num, file_name)

# def find_duplicates(input_file, unique_folder, duplicate_folder):
#     try:
#         start_time = time.time()
#         unique_values = set()
#         value_counts = {}
#         num_lines = 0
        
#         # Get the base name of the file without the extension
#         base_name = os.path.basename(input_file).rsplit('.', 2)[0]

#         # Construct output file paths
#         unique_file_path = os.path.join(unique_folder, f"{base_name}_unq.gen.gz")
#         duplicate_file_path = os.path.join(duplicate_folder, f"{base_name}_dup.gen.gz")

#         with gzip.open(input_file, 'rt') as file:
#             with gzip.open(duplicate_file_path, 'wt') as duplicate_handle:
#                 with gzip.open(unique_file_path, 'wt') as unique_handle:
#                     for line in file:
#                         num_lines += 1
#                         columns = line.split()
#                         column_3_value = columns[2]
#                         value_counts[column_3_value] = value_counts.get(column_3_value, 0) + 1
#                         if value_counts[column_3_value] == 1:
#                             unique_values.add(column_3_value)
#                             unique_handle.write(line)
#                         else:
#                             duplicate_handle.write(line)

#         num_columns = len(columns)
#         print(f"Shape of data in {input_file}: {num_lines} lines x {num_columns} columns")
#         print(f"Number of Unique lines based on column 3: {len(unique_values)}")
#         print(f"Number of duplicate lines based on column 3: {num_lines - len(unique_values)}")

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Execution time for {input_file}: {elapsed_time} seconds")

#     except FileNotFoundError:
#         print(f"File not found: {input_file}. Please provide a valid file path.")

# def process_folder(input_folder, unique_folder, duplicate_folder):
#     # Create output directories if they do not exist
#     os.makedirs(unique_folder, exist_ok=True)
#     os.makedirs(duplicate_folder, exist_ok=True)

#     files = os.listdir(input_folder)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)

#     # Iterate over all files in the input folder
#     for file_name in sorted_files:
#         if file_name.endswith('.gen.gz'):
#             input_file_path = os.path.join(input_folder, file_name)
#             print(f"Processing File:\n {file_name}")
#             find_duplicates(input_file_path, unique_folder, duplicate_folder)

# if __name__ == "__main__":
#     input_folder = args.in_dict
#     unique_folder = args.out_dict_unq
#     duplicate_folder = args.out_dict_dup
#     process_folder(input_folder, unique_folder, duplicate_folder)


#********************* Removing duplicate SNPs from a single file *****************************#
# import gzip
# import os
# import time
# import argparse

# parser = argparse.ArgumentParser(description='Remove duplicate SNP ocurrences from sample files.')
# # parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind', help='Path to the folder with sample_n.gen.gz files')
# # parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
# # parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')

# parser.add_argument('-in_file', type=str, required=True, help='Path to the folder with sample_n.gen.gz files')
# parser.add_argument('-out_d# parser = argparse.ArgumentParser(description='Remove duplicate SNP ocurrences from sample files.')
# # parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind', help='Path to the folder with sample_n.gen.gz files')
# # parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
# # parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')

# parser.add_argument('-in_file', type=str, required=True, help='Path to the folder with sample_n.gen.gz files')
# parser.add_argument('-out_dict_dup', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/duplicates' , help='Path to folder storing duplicate SNPs per sample files')
# parser.add_argument('-out_dict_unq', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/unique' , help='Path to folder storing unique SNPs per sample files')

# args = parser.parse_args()
# def find_duplicates(input_file, unique_folder, duplicate_folder):
#     try:
#         print(f"Processing File:\n {input_file}")
#         start_time = time.time()
#         unique_values = set()
#         value_counts = {}
#         num_lines = 0
#         # Get the base name of the file without the extension
#         base_name = os.path.basename(input_file).rsplit('.', 2)[0]

#         # Construct output file paths
#         unique_file_path = os.path.join(unique_folder, f"{base_name}_unq.gen.gz")
#         duplicate_file_path = os.path.join(duplicate_folder, f"{base_name}_dup.gen.gz")

#         with gzip.open(input_file, 'rt') as file:
#             with gzip.open(duplicate_file_path, 'wt') as duplicate_handle:
#                 with gzip.open(unique_file_path, 'wt') as unique_handle:
#                     for line in file:
#                         #print(f"Line is:\n {line}")
#                         num_lines += 1
#                         columns = line.split()
#                         column_3_value = columns[2]  
#                         value_counts[column_3_value] = value_counts.get(column_3_value, 0) + 1
#                         if value_counts[column_3_value] == 1:
#                             unique_values.add(column_3_value)
#                             unique_handle.write(line)
#                         else:
#                             duplicate_handle.write(line)
    
#         num_columns = len(columns)
#         print("Shape of data in", input_file, ":", num_lines, "lines x", num_columns, "columns")
#         print("Number of Unique lines based on column 3:", len(unique_values))
#         print("Number of duplicate lines based on column 3:", num_lines - len(unique_values))

#         end_time = time.time() 
#         elapsed_time = end_time - start_time
#         print("Execution time:", elapsed_time, "seconds")

#     except FileNotFoundError:
#         print("File not found. Please provide a valid file path.")

# if __name__ == "__main__":
#     input_folder = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind'
#     input_file = os.path.join(input_folder, args.in_file)
#     unique_folder = args.out_dict_unq
#     duplicate_folder = args.out_dict_dup
#     find_duplicates(input_file, unique_folder, duplicate_folder)


