import os
import gzip
import argparse
import multiprocessing as mp
import numpy as np
import gc
import string
from functools import partial

# Argument Parser
parser = argparse.ArgumentParser(description='Appending Genotype Data to Individual Sample Files')
parser.add_argument('-in_file', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/chr18_1a.gen.gz',help='Path to the chr1_3a.gen.gz file')
parser.add_argument('-sample_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_6_processed', help='Path to folder containing sample files')
parser.add_argument('-float_start', type=int, default=52505, help='Start index for float data')
parser.add_argument('-float_end', type=int, default=63005, help='End index for float data')
parser.add_argument('-insert_index', type=int, default=6807475, help='Index to insert new data')
args = parser.parse_args()

def process_sample(sample_file, string_data, float_data, insert_index):
    sample_number = int(os.path.basename(sample_file).split('_')[1].split('.')[0])
    sample_start_index = (sample_number - 17501) * 3
    sample_end_index = sample_start_index + 3

    try:
        with gzip.open(sample_file, 'rt') as f:
            lines = f.readlines()

        new_lines = []
        for i in range(len(string_data)):
            selected_columns = list(string_data[i]) + list(float_data[i][sample_start_index:sample_end_index])
            
            new_lines.append(' '.join(map(str, selected_columns)) + '\n')
        
        #print(f"Number of new lines to be inserted: {len(new_lines)}")
        lines[insert_index:insert_index] = new_lines
        #print(f"New number of lines in {sample_file}: {len(lines)}")

        with gzip.open(sample_file, 'wt') as f:
            f.writelines(lines)

        print(f"Updated {sample_file}")
    except Exception as e:
        print(f"Error processing {sample_file}: {e}")

# def read_genotype_file(gen_file_path, float_start, float_end):
#     try:
#         print(f"Reading genotype file: {gen_file_path}")
#         with gzip.open(gen_file_path, 'rt') as in_handler:
#             content = in_handler.read().splitlines()
        
#         print(f"Number of lines read from input file: {len(content)}")
        
#         string_data = []
#         float_data = []

#         printable = set(string.printable)
        
#         for line_num, line in enumerate(content, 1):
#             # Remove any non-printable characters
#             clean_line = ''.join(filter(lambda x: x in printable, line))
#             parts = clean_line.split()
#             string_data.append(parts[:5])
#             try:
#                 float_values = [np.float16(x) for x in parts[float_start:float_end]]
#                 float_data.append(float_values)
#             except ValueError as e:
#                 print(f"Error on line {line_num}: Could not convert string to float.")
#                 print(f"Problematic data: {parts[float_start:float_end]}")
#                 print(f"Error message: {str(e)}")
#                 continue
        
#         print(f"Successfully processed {len(string_data)} lines of data")
#         return np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
#     except Exception as e:
#         print(f"Error reading genotype file {gen_file_path}: {e}")
#         return None, None

def read_genotype_file(gen_file_path, float_start, float_end):
    try:
        print(f"Reading genotype file: {gen_file_path}")
        string_data = []
        float_data = []

        printable = set(string.printable)
        
        with gzip.open(gen_file_path, 'rt') as in_handler:
            for line_num, line in enumerate(in_handler, 1):
                
                # Remove any non-printable characters
                clean_line = ''.join(filter(lambda x: x in printable, line.strip()))
                parts = clean_line.split()
                
                string_data.append(parts[:5])
                try:
                    float_values = [np.float16(x) for x in parts[float_start:float_end]]
                    float_data.append(float_values)
                except ValueError as e:
                    print(f"Error on line {line_num}: Could not convert string to float.")
                    print(f"Problematic data: {parts[float_start:float_end]}")
                    print(f"Error message: {str(e)}")
                    continue
                print(f"Processed line:{line_num}\n")
                # Clear the lists periodically to free up memory
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} lines")
                    gc.collect()  # Force garbage collection
        
        print(f"Successfully processed {len(string_data)} lines of data")
        return np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
    except Exception as e:
        print(f"Error reading genotype file {gen_file_path}: {e}")
        return None, None
  
def process_genotype_file(gen_file, sample_folder, float_start, float_end, insert_index):
    # Get all sample files
    sample_files = sorted(
    [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.endswith('.gen.gz')],
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)
    
    print(f"Number of sample files to process is: {len(sample_files)}")

    #sample_files = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.endswith('.gen.gz')]

    # Read the entire chr1_1a.gen.gz file
    string_data, float_data = read_genotype_file(gen_file, float_start, float_end)
    if string_data is None or float_data is None:
        print("Failed to read input file. Exiting.")
        return

    # Use multiprocessing to update sample files
    print("Starting multiprocessing pool")
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     pool.map(process_sample, [(sample_file, string_data, float_data, insert_index) for sample_file in sample_files])

    process_func = partial(process_sample, string_data=string_data, float_data=float_data, insert_index=insert_index)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_func, sample_files)
    
    print("All sample files processed")


if __name__ == "__main__":
    in_file = args.in_file
    sample_folder = args.sample_folder
    float_start = args.float_start
    float_end = args.float_end
    insert_index = args.insert_index

    process_genotype_file(in_file, sample_folder, float_start, float_end, insert_index)


