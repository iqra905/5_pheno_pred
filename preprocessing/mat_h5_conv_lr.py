#*************************** Process the per sample file chromosome wise *******************************#
import os
import scipy.io
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import argparse


parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-base_input_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq', help='Path to the folder containing .mat sample files')
parser.add_argument('-output_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/combined_h5_sample_1', help='Path to the output folder')
parser.add_argument('-metadata_file', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.gen', help='Path to the metadata file')

args = parser.parse_args()

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[1])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def load_metadata_for_chromosome(metadata_file, chromosome):
    columns = ['chromosome', 'snp_id', 'position', 'ref', 'alt']
    df = pd.read_csv(metadata_file, sep=r'\s+', header=None, names=columns)
    chr_data = df[df['chromosome'] == chromosome].reset_index(drop=True)
    #print(f"Loaded metadata for chromosome {chromosome}:")
    #print(chr_data.head())
    #print(f"Total SNPs in metadata for chromosome {chromosome}: {len(chr_data)}")
    return chr_data

def convert_genotype(genotype, ref, alt):
    if genotype == f"{ref}{ref}":
        return 0
    elif genotype in [f"{ref}{alt}", f"{alt}{ref}"]:
        return 1
    elif genotype == f"{alt}{alt}":
        return 2
    else:
        return -1

def process_chromosome_folder(args):
    input_folder, output_file, metadata_file, chromosome = args
    print(f"\nProcessing chromosome {chromosome}")
    
    chr_metadata = load_metadata_for_chromosome(metadata_file, chromosome)
    
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    mat_files = sorted(mat_files, key=custom_sort)
    
    if not mat_files:
        print(f"No .mat files found in {input_folder}")
        return
    
    first_file_path = os.path.join(input_folder, mat_files[0])
    first_file = scipy.io.loadmat(first_file_path)
    
    genotype_string = first_file['genotypes'][0]
    genotypes = genotype_string.split()
    n_snps = len(genotypes)
    
    if n_snps != len(chr_metadata):
        print(f"Warning: Number of SNPs in .mat files ({n_snps}) does not match metadata ({len(chr_metadata)})")
    
    n_samples = len(mat_files)
    
    combined_data = np.full((n_samples, n_snps), -1, dtype=np.int32)
    
    print(f"Processing {n_samples} samples for {n_snps} SNPs")
    
    for i, mat_file in enumerate(tqdm(mat_files, desc=f"Processing samples for chromosome {chromosome}")):
        mat_data = scipy.io.loadmat(os.path.join(input_folder, mat_file))
        genotype_string = mat_data['genotypes'][0]
        genotypes = genotype_string.split()
        
        for j, genotype in enumerate(genotypes):
            if j < len(chr_metadata):
                ref, alt = chr_metadata.loc[j, ['ref', 'alt']]
                combined_data[i, j] = convert_genotype(genotype, ref, alt)
            else:
                combined_data[i, j] = -1
    
    print(f"\nSample of combined numerical data for chromosome {chromosome} (first 5 rows, first 5 columns):")
    print(combined_data[:5, :5])
    
    unique, counts = np.unique(combined_data, return_counts=True)
    print(f"\nValue counts in combined data for chromosome {chromosome}:")
    for value, count in zip(unique, counts):
        print(f"{value}: {count}")
    
    with h5py.File(output_file, 'w') as f:
        dataset = f.create_dataset('data', data=combined_data, 
                                   dtype='<i4',
                                   compression="gzip",
                                   compression_opts=9,
                                   shuffle=True,
                                   fletcher32=True)
        
        dataset.attrs['CLASS'] = np.string_("EARRAY")
        dataset.attrs['EXTDIM'] = 0
        dataset.attrs['TITLE'] = np.string_(f"chromosome_{chromosome}")
        dataset.attrs['VERSION'] = np.string_("1.1")
    
    print(f"\nSaved combined data for chromosome {chromosome} to {output_file}")

def main(base_input_folder, output_folder, metadata_file):

    print(f"Base Input folder: {base_input_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Reference file: {metadata_file}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    args_list = []
    for chromosome in range(1, 7):  # Process chromosomes 1 to 22
        input_folder = os.path.join(base_input_folder, f"chromosome_{chromosome}")
        output_file = os.path.join(output_folder, f"t2d_chr{chromosome}.h5")
        
        if os.path.exists(input_folder):
            args_list.append((input_folder, output_file, metadata_file, chromosome))
        else:
            print(f"Folder for chromosome {chromosome} not found: {input_folder}")
    
    # Use multiprocessing to process chromosomes in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_chromosome_folder, args_list)

if __name__ == "__main__":
   main(args.base_input_folder, args.output_folder, args.metadata_file)

# import os
# import scipy.io
# import h5py
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# def load_metadata_for_chromosome(metadata_file, chromosome):
#     columns = ['chromosome', 'snp_id', 'position', 'ref', 'alt']
#     df = pd.read_csv(metadata_file, sep=r'\s+', header=None, names=columns)
#     chr_data = df[df['chromosome'] == chromosome].reset_index(drop=True)
#     print(f"Loaded metadata for chromosome {chromosome}:")
#     print(chr_data.head())
#     print(f"Total SNPs in metadata for chromosome {chromosome}: {len(chr_data)}")
#     return chr_data

# def convert_genotype(genotype, ref, alt):
#     if genotype == f"{ref}{ref}":
#         return 0
#     elif genotype in [f"{ref}{alt}", f"{alt}{ref}"]:
#         return 1
#     elif genotype == f"{alt}{alt}":
#         return 2
#     else:
#         return -1

# def process_chromosome_folder(input_folder, output_file, metadata_file, chromosome):
#     print(f"\nProcessing chromosome {chromosome}")
    
#     chr_metadata = load_metadata_for_chromosome(metadata_file, chromosome)
    
#     # Get list of all .mat files in the folder
#     mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    
#     if not mat_files:
#         print(f"No .mat files found in {input_folder}")
#         return
    
#     # Load the first file to get dimensions
#     first_file_path = os.path.join(input_folder, mat_files[0])
#     first_file = scipy.io.loadmat(first_file_path)
    
#     genotype_string = first_file['genotypes'][0]
#     genotypes = genotype_string.split()
#     n_snps = len(genotypes)
    
#     if n_snps != len(chr_metadata):
#         print(f"Warning: Number of SNPs in .mat files ({n_snps}) does not match metadata ({len(chr_metadata)})")
    
#     n_samples = len(mat_files)
    
#     # Initialize the combined data array
#     combined_data = np.full((n_samples, n_snps), -1, dtype=np.int32)
    
#     print(f"Processing {n_samples} samples for {n_snps} SNPs")
    
#     # Process each sample file
#     for i, mat_file in enumerate(tqdm(mat_files, desc="Processing samples")):
#         mat_data = scipy.io.loadmat(os.path.join(input_folder, mat_file))
#         genotype_string = mat_data['genotypes'][0]
#         genotypes = genotype_string.split()
        
#         for j, genotype in enumerate(genotypes):
#             if j < len(chr_metadata):
#                 ref, alt = chr_metadata.loc[j, ['ref', 'alt']]
#                 combined_data[i, j] = convert_genotype(genotype, ref, alt)
#             else:
#                 combined_data[i, j] = -1
    
#     print("\nSample of combined numerical data (first 5 rows, first 5 columns):")
#     print(combined_data[:5, :5])
    
#     unique, counts = np.unique(combined_data, return_counts=True)
#     print("\nValue counts in combined data:")
#     for value, count in zip(unique, counts):
#         print(f"{value}: {count}")
    
#     with h5py.File(output_file, 'w') as f:
#         dataset = f.create_dataset('data', data=combined_data, 
#                                    dtype='<i4',
#                                    compression="gzip",
#                                    compression_opts=9,
#                                    shuffle=True,
#                                    fletcher32=True)
        
#         dataset.attrs['CLASS'] = np.string_("EARRAY")
#         dataset.attrs['EXTDIM'] = 0
#         dataset.attrs['TITLE'] = np.string_(f"chromosome_{chromosome}")
#         dataset.attrs['VERSION'] = np.string_("1.1")
    
#     print(f"\nSaved combined data to {output_file}")

# def main():
#     base_input_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/combined_h5_sample"
#     metadata_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.gen"  
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     for chromosome in range(1, 23):  # Process chromosomes 1 to 22
#         input_folder = os.path.join(base_input_folder, f"chromosome_{chromosome}")
#         output_file = os.path.join(output_folder, f"t2d_chr{chromosome}.h5")
        
#         if os.path.exists(input_folder):
#             process_chromosome_folder(input_folder, output_file, metadata_file, chromosome)
#         else:
#             print(f"Folder for chromosome {chromosome} not found: {input_folder}")

# if __name__ == "__main__":
#     main()

#*************************** Process the entire combined sample file chromosome wise *******************************#
# import os
# import scipy.io
# import h5py
# import pandas as pd
# import numpy as np

# def load_metadata_for_chromosome(metadata_file, chromosome):
#     columns = ['chromosome', 'snp_id', 'position', 'ref', 'alt']
#     df = pd.read_csv(metadata_file, sep=r'\s+', header=None, names=columns)
#     chr_data = df[df['chromosome'] == chromosome].reset_index(drop=True)
#     print(f"Loaded metadata for chromosome {chromosome}:")
#     print(chr_data.head())
#     print(f"Total SNPs in metadata for chromosome {chromosome}: {len(chr_data)}")
#     return chr_data

# def convert_genotype(genotype, ref, alt):
#     if pd.isna(genotype) or pd.isna(ref) or pd.isna(alt):
#         return -1
#     if genotype == f"{ref}{ref}":
#         return 0
#     elif genotype in [f"{ref}{alt}", f"{alt}{ref}"]:
#         return 1
#     elif genotype == f"{alt}{alt}":
#         return 2
#     else:
#         return -1

# def process_mat_file(input_file, output_file, metadata_file):
#     chr_num = int(input_file.split('chr')[1].split('.')[0])
#     print(f"\nProcessing chromosome {chr_num}")
    
#     chr_metadata = load_metadata_for_chromosome(metadata_file, chr_num)
    
#     mat_data = scipy.io.loadmat(input_file)
#     data = mat_data['combined_genotypes']
#     print(f"Shape of data in .mat file: {data.shape}")
#     print("Sample of .mat file data (first 5 rows, first 5 columns):")
#     print(data[:5, :5])
    
#     if data.shape[1] != len(chr_metadata):
#         print(f"Warning: Number of SNPs in .mat file ({data.shape[1]}) does not match metadata ({len(chr_metadata)})")
    
#     numerical_data = np.full(data.shape, -1, dtype=np.int32)
    
#     print("\nConverting genotypes:")
#     for i in range(data.shape[1]):
#         if i < len(chr_metadata):
#             ref, alt = chr_metadata.loc[i, ['ref', 'alt']]
#             numerical_data[:, i] = [convert_genotype(g, ref, alt) for g in data[:, i]]
#         else:
#             print(f"Warning: No metadata for column {i}, setting all values to -1")
#             numerical_data[:, i] = -1
        
#         if i < 5:  # Print detailed info for first 5 SNPs
#             print(f"\nSNP {i}:")
#             print(f"Ref: {ref}, Alt: {alt}")
#             print("Original data (first 5 samples):", data[:5, i])
#             print("Converted data (first 5 samples):", numerical_data[:5, i])
        
#         if i % 1000 == 0:  # Print progress every 1000 SNPs
#             print(f"Processed {i} SNPs...")
    
#     print("\nSample of converted numerical data (first 5 rows, first 5 columns):")
#     print(numerical_data[:5, :5])
    
#     unique, counts = np.unique(numerical_data, return_counts=True)
#     print("\nValue counts in converted data:")
#     for value, count in zip(unique, counts):
#         print(f"{value}: {count}")
    
#     with h5py.File(output_file, 'w') as f:
#         dataset = f.create_dataset('data', data=numerical_data, 
#                                    dtype='<i4',
#                                    compression="gzip",
#                                    compression_opts=9,
#                                    shuffle=True,
#                                    fletcher32=True)
        
#         dataset.attrs['CLASS'] = np.string_("EARRAY")
#         dataset.attrs['EXTDIM'] = 0
#         dataset.attrs['TITLE'] = np.string_(os.path.basename(input_file).replace('.mat', ''))
#         dataset.attrs['VERSION'] = np.string_("1.1")
    
#     print(f"\nSaved converted data to {output_file}")

# def main():
#     input_folder ="/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/combined"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/combined_h5"
#     metadata_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.gen"  
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.mat') and filename.startswith('t2d_chr'):
#             input_file = os.path.join(input_folder, filename)
#             output_file = os.path.join(output_folder, filename.replace('.mat', '.h5'))
#             process_mat_file(input_file, output_file, metadata_file)
#             print(f"Processed {filename}")

# if __name__ == "__main__":
#     main()