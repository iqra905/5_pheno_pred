import os
import gzip
import argparse
import time
import multiprocessing as mp

# Argument Parser
parser = argparse.ArgumentParser(description='Generating Genotype Files per Individual')
parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen', help='Path to the folder with .gen.gz files')
parser.add_argument('-out_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_samplefin', help='Path to folder storing per sample files')
parser.add_argument('-sample_count', type=int, default=37663, help='Total number of samples.')

args = parser.parse_args()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def process_sample(sample_idx, gen_file, output_folder):
    output_file_path = os.path.join(output_folder, f"sample_{sample_idx:05}.gen.gz")
    sample_start_index = 5 + (sample_idx - 1) * 3
    sample_end_index = sample_start_index + 3
    num_snps = 0

    try:
        with gzip.open(os.path.join(input_folder, gen_file), 'rt') as in_handler:
            with gzip.open(output_file_path, 'at') as out_handler:
                for line in in_handler:
                    parts = line.split()
                    selected_columns = parts[:5] + parts[sample_start_index:sample_end_index]
                    selected_line = ' '.join(selected_columns)
                    out_handler.write(selected_line + '\n')
                    num_snps += 1
        print(f"Sample {sample_idx:05}: Added {num_snps} SNPs to {output_file_path}")
    except Exception as e:
        print(f"Error processing sample {sample_idx:05}: {e}")

def process_genotype_file(input_folder, output_folder, sample_count):
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
    print(f"Found {len(sorted_files)} genotype files.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    for gen_file in sorted_files:
        print(f"Processing file: {gen_file}")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(
                process_sample,
                [(i, gen_file, output_folder) for i in range(1, sample_count + 1)]
            )
        print(f"Finished processing file: {gen_file}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    input_folder = args.in_dict
    output_folder = args.out_dict
    sample_count = args.sample_count

    process_genotype_file(input_folder, output_folder, sample_count)
