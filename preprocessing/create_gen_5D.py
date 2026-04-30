import gzip
import glob
import os
import itertools
import argparse
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-input_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm', help='Path to the folder containing .gen.gz files')
parser.add_argument('-reference_file', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can.gen', help='Path to the reference file')
parser.add_argument('-output_file', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/pros/pros_can_merged.gen_111.gz', help='Path to the output file')
parser.add_argument('-chunk_size', type=int, default=10000, help='Number of SNPs to process at a time.')

args = parser.parse_args()


def read_reference_file(filename):
    with open(filename, 'r') as f:
        return {tuple(line.split()[:5]) for line in f}

def process_chunk(args):
    chunk, reference_set = args
    return [line for line in chunk if tuple(line.split()[:5]) in reference_set]

def process_gen_file(input_folder, gen_file, reference_set, output_file, chunk_size):
   
    gen_file_path = os.path.join(input_folder, gen_file)
    print(f"Processing {gen_file_path}...")
    with gzip.open(gen_file_path, 'rt') as in_f, gzip.open(output_file, 'at') as out_f:
        pool = Pool(processes=cpu_count())
        chunk_generator = iter(lambda: list(itertools.islice(in_f, chunk_size)), [])
        for filtered_chunk in pool.imap(process_chunk, ((chunk, reference_set) for chunk in chunk_generator)):
            out_f.writelines(filtered_chunk)
        pool.close()
        pool.join()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def main(input_folder, reference_file, output_file, chunk_size):
    print(f"Input folder: {input_folder}")
    print(f"Reference file: {reference_file}")
    print(f"Output file: {output_file}")

    print("Reading reference file...")
    reference_set = read_reference_file(reference_file)

    print("Processing .gen.gz files...")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied for accessing '{input_folder}'.")
        return

    gen_files = [f for f in files if f.endswith('.gen.gz')]
    gen_files = sorted(gen_files, key=custom_sort)
    
    if not gen_files:
        print(f"No .gen.gz files found in {input_folder}")
        print("Files in directory:")
        print("\n".join(os.listdir(input_folder)))
        return

    # Ensure the output file is empty before we start
    open(output_file, 'w').close()

    for gen_file in gen_files:
        process_gen_file(input_folder,gen_file, reference_set, output_file, chunk_size)

    print(f"Merged and filtered file created: {output_file}")

if __name__ == "__main__":
    
    main(args.input_folder, args.reference_file, args.output_file, args.chunk_size)