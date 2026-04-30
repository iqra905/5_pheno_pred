import os
import gzip
import argparse
import time
import multiprocessing as mp
import numpy as np
import psutil
import gc
from functools import partial

# Argument Parser
parser = argparse.ArgumentParser(description='Generating Genotype Files per Individual')
parser.add_argument('-in_dict', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm', help='Path to the folder with .gen.gz files')
parser.add_argument('-out_dict', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data', help='Path to folder storing per sample files')
parser.add_argument('-sample_count', type=int, default=3500, help='Total number of samples.')
parser.add_argument('-sample_idx_add', type=int, default=0, help='Sample Idx name for file write.')
parser.add_argument('-chr_start', type=int, default=1, help='Start chromosome number.')
parser.add_argument('-chr_end', type=int, default=7, help='End chromosome number.')
parser.add_argument('-float_start', type=int, default=5, help='Start index for float data.')
parser.add_argument('-float_end', type=int, default=22505, help='End index for float data.')




args = parser.parse_args()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def print_memory_usage(message):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{message} - Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def process_sample(sample_idx, string_data, float_data, output_folder, chunk_num, gen_file, sample_idx_add):
    sample_idx_n = sample_idx + sample_idx_add
    output_file_path = os.path.join(output_folder, f"sample_{sample_idx_n:05}.gen.gz")
    sample_start_index = (sample_idx - 1) * 3
    sample_end_index = sample_start_index + 3
    num_snps = 0

    try:
        with gzip.open(output_file_path, 'at') as out_handler:
            for i in range(len(string_data)):
                selected_columns = list(string_data[i]) + list(float_data[i][sample_start_index:sample_end_index])
                selected_line = ' '.join(map(str, selected_columns))
                out_handler.write(selected_line + '\n')
                num_snps += 1
        #print(f"Sample {sample_idx_n:05}, Chunk {chunk_num:03} of input file {gen_file}: Added {num_snps} SNPs to {output_file_path}")
    except Exception as e:
        print(f"Error processing sample {sample_idx_n:05}, Chunk {chunk_num:03} of input file {gen_file}: {e}")

def read_genotype_file_in_chunks(gen_file_path, float_start, float_end, chunk_size=50000):
    try:
        with gzip.open(gen_file_path, 'rt') as in_handler:
            string_data = []
            float_data = []
            chunk_num = 0
            #print(f"Loading chunk number {chunk_num} of file {gen_file_path} \n")
            for line_idx, line in enumerate(in_handler):
                parts = line.split()
                string_data.append(parts[:5])
                float_data.append([np.float16(x) for x in parts[float_start:float_end]])
                
                if (line_idx + 1) % chunk_size == 0:
                    chunk_num += 1
                    yield chunk_num, np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
                    string_data = []
                    float_data = []
            
            if string_data:
                chunk_num += 1
                yield chunk_num, np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
    except Exception as e:
        print(f"Error reading genotype file {gen_file_path}: {e}")

def process_genotype_file(input_folder, output_folder, sample_count, chr_start, chr_end, float_start, float_end, sample_idx_add):
    start_time = time.time()

    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied for accessing '{input_folder}'.")
        return

    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)

    # Filter files based on the specified chromosome range
    filtered_files = [f for f in sorted_files if chr_start <= int(f.split('_')[0][3:]) <= chr_end]

    print(f"Sorted and Filtered Files are: \n{filtered_files}")
    print(f"Found {len(filtered_files)} genotype files within chromosome range {chr_start}-{chr_end}.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for gen_file in filtered_files:
        print(f"Processing file: {gen_file}")
        gen_file_path = os.path.join(input_folder, gen_file)

        for chunk_num, string_data, float_data in read_genotype_file_in_chunks(gen_file_path,float_start, float_end):
            print(f"Processing chunk {chunk_num:03} of file: {gen_file}")

            # Use partial to pre-fill the arguments except for sample_idx
            partial_process_sample = partial(process_sample, string_data=string_data, float_data=float_data, output_folder=output_folder, chunk_num=chunk_num, gen_file=gen_file, sample_idx_add=sample_idx_add)
            
            with mp.Pool(processes=mp.cpu_count()) as pool:
                pool.map(partial_process_sample, range(1, sample_count + 1))

            del string_data, float_data
            gc.collect()
        
        print(f"Finished processing file: {gen_file}")
        print_memory_usage(f"After processing {gen_file}.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print_memory_usage("At the end of processing")

if __name__ == "__main__":
    input_folder = args.in_dict
    output_folder = args.out_dict
    sample_count = args.sample_count
    chr_start = args.chr_start
    chr_end = args.chr_end
    float_start = args.float_start
    float_end = args.float_end
    sample_idx_add = args.sample_idx_add

    process_genotype_file(input_folder, output_folder, sample_count, chr_start, chr_end, float_start, float_end, sample_idx_add)


# import os
# import gzip
# import argparse
# import time
# import multiprocessing as mp
# import numpy as np
# import psutil
# import gc



# # Argument Parser
# parser = argparse.ArgumentParser(description='Generating Genotype Files per Individual')
# parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen', help='Path to the folder with .gen.gz files')
# parser.add_argument('-out_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_sample_fin', help='Path to folder storing per sample files')
# parser.add_argument('-sample_count', type=int, default=3000, help='Total number of samples.')

# args = parser.parse_args()

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])
#     except ValueError:
#         chr_num = float('inf')
#     return (chr_num, file_name)

# def print_memory_usage(message):
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     print(f"{message} - Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")

# def process_sample(sample_idx, string_data, float_data, output_folder, chunk_num):
#     sample_idx_n = sample_idx + 3000
#     output_file_path = os.path.join(output_folder, f"sample_{sample_idx_n:05}.gen.gz")
#     sample_start_index = (sample_idx - 1) * 3
#     sample_end_index = sample_start_index + 3
#     num_snps = 0

#     try:
#         with gzip.open(output_file_path, 'at') as out_handler:
#             for i in range(len(string_data)):
#                 selected_columns = list(string_data[i]) + list(float_data[i][sample_start_index:sample_end_index])
#                 selected_line = ' '.join(map(str, selected_columns))
#                 out_handler.write(selected_line + '\n')
#                 num_snps += 1
#         print(f"Sample {sample_idx_n:05}, Chunk {chunk_num:03}: Added {num_snps} SNPs to {output_file_path}")
#     except Exception as e:
#         print(f"Error processing sample {sample_idx:05}, Chunk {chunk_num:03}: {e}")


# def read_genotype_file_in_chunks(gen_file_path, chunk_size=80000):
#     try:
#         with gzip.open(gen_file_path, 'rt') as in_handler:
#             string_data = []
#             float_data = []
#             chunk_num = 0
#             for line_idx, line in enumerate(in_handler):
#                 parts = line.split()
#                 string_data.append(parts[:5])
#                 float_data.append([np.float16(x) for x in parts[9005:18005]])
#                 #print(f"Float Data is:\n {float_data}")
                
#                 if (line_idx + 1) % chunk_size == 0:
#                     chunk_num += 1
#                     yield chunk_num, np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
#                     string_data = []
#                     float_data = []
            
#             if string_data:
#                 chunk_num += 1
#                 yield chunk_num, np.array(string_data, dtype=object), np.array(float_data, dtype=np.float16)
#     except Exception as e:
#         print(f"Error reading genotype file {gen_file_path}: {e}")

# def process_genotype_file(input_folder, output_folder, sample_count):
#     start_time = time.time()

#     try:
#         files = os.listdir(input_folder)
#     except FileNotFoundError:
#         print(f"Error: Input folder '{input_folder}' not found.")
#         return
#     except PermissionError:
#         print(f"Error: Permission denied for accessing '{input_folder}'.")
#         return

#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
#     print(f"Found {len(sorted_files)} genotype files.")

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for gen_file in sorted_files:
#         print(f"Processing file: {gen_file}")
#         gen_file_path = os.path.join(input_folder, gen_file)

#         for chunk_num, string_data, float_data in read_genotype_file_in_chunks(gen_file_path):
#             print(f"Processing chunk {chunk_num} of file: {gen_file}")
#             #print_memory_usage(f"After reading chunk {chunk_num} of genotype file")
            
#             # with mp.Manager() as manager:
#             #     shared_string_data = manager.list(string_data)
#             #     shared_float_data = manager.list(float_data)
#             #     del string_data, float_data  # Free memory
#             #     gc.collect()  # Force garbage collection

#             #     print_memory_usage("After creating shared lists")

#             with mp.Pool(processes=mp.cpu_count()) as pool:
#                 pool.starmap(
#                     process_sample,
#                     [(i, string_data, float_data, output_folder, chunk_num) for i in range(1, sample_count + 1)]
#                 )

#             #print_memory_usage(f"After processing chunk {chunk_num} of samples")
#             del string_data, float_data
#             gc.collect()
        
#         print(f"Finished processing file: {gen_file}")
#         print_memory_usage(f"After processing {gen_file}.")

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Execution time: {elapsed_time:.2f} seconds")
#     print_memory_usage("At the end of processing")

# if __name__ == "__main__":
#     input_folder = args.in_dict
#     output_folder = args.out_dict
#     sample_count = args.sample_count

#     process_genotype_file(input_folder, output_folder, sample_count)
    
