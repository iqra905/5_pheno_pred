import os
import gzip
import multiprocessing as mp
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description='Generating Genotype Files per Individual')
parser.add_argument('-in_dict', type=str, default='/vol/research/ucdatasets/gwas/sampled_data', help='Path to the folder with .gen.gz files')
parser.add_argument('-out_dict', type=str, default='/vol/research/ucdatasets/gwas/sampled_data/s1', help='Path to folder storing per sample files')
# parser.add_argument('-in_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/samples', help='Path to the folder with .gen.gz files')
# parser.add_argument('-out_dict', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/samples/samples_processed', help='Path to folder storing per sample files')

args = parser.parse_args()

def process_file(args):
    file_path, output_folder = args
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    #print(f"Processing {file_path}")
    with gzip.open(file_path, 'rt') as infile, gzip.open(output_file_path, 'wt') as outfile:
        for i, line in enumerate(infile):
            #print(f"Processing line {i} in {file_path}")
            if i >= 13832663:
                break
            outfile.write(line)
        print(f"Processed: {file_path}\n")

def main(input_folder, output_folder):
    # Get all .gen.gz files and sort them by name
    #all_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')])
    #file_paths = [(os.path.join(input_folder, f), output_folder) for f in all_files]

    sample_files = [f'sample_{i:05d}.gen.gz' for i in range(1, 6001)]
    file_paths = [(os.path.join(input_folder, f), output_folder) for f in sample_files]
    print(f"Found {len(sample_files)} .gen.gz files in {input_folder}\n")
    print(f"File names under process: {sample_files[0]} to {sample_files[-1]}\n")


    # Use multiprocessing to process files in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_file, file_paths)

if __name__ == "__main__":
    input_folder = args.in_dict
    output_folder = args.out_dict
    os.makedirs(output_folder, exist_ok=True)
    main(input_folder, output_folder)