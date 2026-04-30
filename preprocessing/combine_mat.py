# ******************************* Final script for all chromosomes in parallel ***************************#
import os
import numpy as np
from scipy import io
import glob
import re
import h5py
import multiprocessing
from functools import partial
import argparse

parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-base_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros_seq/pros_seq', help='Path to the folder containing .mat sample files')
parser.add_argument('-output_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros_seq/combined', help='Path to the output folder')
parser.add_argument('-num_processes', type=int, default=8, help='Maximum number of processes to use')
args = parser.parse_args()


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def combine_mat_files(input_folder, output_file, chunk_size=50):
    print(f"Reading .mat files from: {input_folder}")
    mat_files = sorted(glob.glob(os.path.join(input_folder, 'sample_*.mat')), key=natural_sort_key)
    
    if not mat_files:
        print("No .mat files found in the input folder.")
        return

    total_samples = len(mat_files)
    print(f"Found {total_samples} .mat files.")

    # Read the first file to get the number of SNPs
    first_sample = io.loadmat(mat_files[0])['genotypes'][0]
    num_snps = len(first_sample.split())

    print(f"Number of SNPs: {num_snps}")

    # Create an HDF5 file for intermediate storage
    h5_filename_path = output_file + '.h5'
    h5_filename = h5_filename_path.replace(".mat.h5", ".h5")
    with h5py.File(h5_filename, 'w') as h5f:
        dset = h5f.create_dataset('combined_genotypes', shape=(total_samples, num_snps),
                                  dtype='S2', chunks=True, maxshape=(None, num_snps))

        # Process files in chunks
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_files = mat_files[start_idx:end_idx] 
            
            chunk_data = np.empty((len(chunk_files), num_snps), dtype='S2')
            
            for i, file in enumerate(chunk_files):
                try:
                    mat_contents = io.loadmat(file)
                    genotypes = mat_contents['genotypes'][0].split()
                    chunk_data[i, :] = genotypes
                    #print(f"Processed file {start_idx+i+1}/{total_samples}: {os.path.basename(file)}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
            
            dset[start_idx:end_idx, :] = chunk_data
            print(f"Saved chunk: {start_idx+1}-{end_idx}/{total_samples}")

    # print("Converting HDF5 to .mat file...")
    # with h5py.File(h5_filename, 'r') as h5f:
    #     data = h5f['combined_genotypes']
    #     io.savemat(output_file, {'combined_genotypes': data[:]}, do_compression=True)
    #     print(f"Saved all data to .mat file")

    # # Remove the temporary HDF5 file
    # #os.remove(h5_filename)

    # # Verify the shape of the saved .mat file
    # mat_contents = io.whosmat(output_file)
    # if mat_contents:
    #     var_name, shape, _ = mat_contents[0]
    #     print(f"Combined data saved to {output_file}")
    #     print(f"Shape of combined data: {shape}")
    #     if shape[0] != total_samples or shape[1] != num_snps:
    #         print(f"Warning: Shape mismatch. Expected {(total_samples, num_snps)}, got {shape}")
    # else:
    #     print(f"Error: Unable to read shape information from {output_file}")

def process_chromosome(base_folder, output_folder, chrom):
    input_folder = os.path.join(base_folder, f'chromosome_{chrom}')
    output_file = os.path.join(output_folder, f'chr{chrom}.mat')
    
    print(f"\nProcessing Chromosome {chrom}")
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")
    
    if os.path.exists(input_folder):
        combine_mat_files(input_folder, output_file)
    else:
        print(f"Folder for chromosome {chrom} not found: {input_folder}")

def process_all_chromosomes(base_folder, output_folder, num_processes=4):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    num_processes = min(num_processes if num_processes else multiprocessing.cpu_count() - 1, multiprocessing.cpu_count() - 1)

    print(f"Processing with {num_processes} processes")
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Prepare the partial function with fixed arguments
    process_func = partial(process_chromosome, base_folder, output_folder)

    # Process chromosomes in parallel
    pool.map(process_func, range(1, 23))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

def main(base_folder,output_folder,num_processes=None):
    process_all_chromosomes(base_folder, output_folder, num_processes)

if __name__ == "__main__":
    main(args.base_folder,
         args.output_folder,
         args.num_processes)

# ******************************* Final script for chr 1 SNPs ***************************#
# import os
# import numpy as np
# from scipy import io
# import glob
# import re
# import h5py

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# def combine_mat_files(input_folder, output_file, chunk_size=100):
#     print(f"Reading .mat files from: {input_folder}")
#     mat_files = sorted(glob.glob(os.path.join(input_folder, 'sample_*.mat')), key=natural_sort_key)
    
#     if not mat_files:
#         print("No .mat files found in the input folder.")
#         return

#     total_samples = len(mat_files)
#     print(f"Found {total_samples} .mat files.")

#     # Read the first file to get the number of SNPs
#     first_sample = io.loadmat(mat_files[0])['genotypes'][0]
#     num_snps = len(first_sample.split())

#     print(f"Number of SNPs: {num_snps}")

#     # Create an HDF5 file for intermediate storage
#     h5_filename = output_file + '.h5'
#     with h5py.File(h5_filename, 'w') as h5f:
#         dset = h5f.create_dataset('combined_genotypes', shape=(total_samples, num_snps),
#                                   dtype='S2', chunks=True, maxshape=(None, num_snps))

#         # Process files in chunks
#         for start_idx in range(0, total_samples, chunk_size):
#             end_idx = min(start_idx + chunk_size, total_samples)
#             chunk_files = mat_files[start_idx:end_idx]
            
#             chunk_data = np.empty((len(chunk_files), num_snps), dtype='S2')
            
#             for i, file in enumerate(chunk_files):
#                 try:
#                     mat_contents = io.loadmat(file)
#                     genotypes = mat_contents['genotypes'][0].split()
#                     chunk_data[i, :] = genotypes
#                     print(f"Processed file {start_idx+i+1}/{total_samples}: {os.path.basename(file)}")
#                 except Exception as e:
#                     print(f"Error processing file {file}: {e}")
            
#             dset[start_idx:end_idx, :] = chunk_data
#             print(f"Saved chunk: {start_idx+1}-{end_idx}/{total_samples}")

#     print("Converting HDF5 to .mat file...")
#     with h5py.File(h5_filename, 'r') as h5f:
#         data = h5f['combined_genotypes']
#         io.savemat(output_file, {'combined_genotypes': data[:]}, do_compression=True)
#         print(f"Saved all data to .mat file")

#     # Remove the temporary HDF5 file
#     os.remove(h5_filename)

#     # Verify the shape of the saved .mat file
#     mat_contents = io.whosmat(output_file)
#     if mat_contents:
#         var_name, shape, _ = mat_contents[0]
#         print(f"Combined data saved to {output_file}")
#         print(f"Shape of combined data: {shape}")
#         if shape[0] != total_samples or shape[1] != num_snps:
#             print(f"Warning: Shape mismatch. Expected {(total_samples, num_snps)}, got {shape}")
#     else:
#         print(f"Error: Unable to read shape information from {output_file}")

# def main():
#     input_folder = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/chromosome_2'
#     output_file = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq/t2d_chr2.mat'
#     combine_mat_files(input_folder, output_file)

# if __name__ == "__main__":
#     main()

# ******************************* Final script for chr 1 SNPs ***************************#
# import os
# import numpy as np
# from scipy import io
# import glob
# import re

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# def combine_mat_files(input_folder, output_file, chunk_size=100):
#     print(f"Reading .mat files from: {input_folder}")
#     mat_files = sorted(glob.glob(os.path.join(input_folder, 'sample_*.mat')), key=natural_sort_key)
    
#     if not mat_files:
#         print("No .mat files found in the input folder.")
#         return

#     print(f"Found {len(mat_files)} .mat files.")

#     # Read the first file to get the number of SNPs
#     first_sample = io.loadmat(mat_files[0])['genotypes'][0]
#     num_snps = len(first_sample.split())

#     total_samples = len(mat_files)
    
#     # Create a memory-mapped array for the output
#     mmap_filename = output_file + '.mmap'
#     mmap_array = np.memmap(mmap_filename, dtype='U2', mode='w+', shape=(total_samples, num_snps))

#     # Process files
#     for i, file in enumerate(mat_files):
#         try:
#             mat_contents = io.loadmat(file)
#             genotypes = mat_contents['genotypes'][0].split()
#             mmap_array[i, :] = genotypes
#             print(f"Processed file {i+1}/{total_samples}: {os.path.basename(file)}")
            
#             # Flush to disk every chunk_size samples
#             if (i + 1) % chunk_size == 0 or i == total_samples - 1:
#                 mmap_array.flush()
#                 print(f"Flushed chunk to disk: {i+1}/{total_samples}")
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")

#     # Close the memmap array
#     del mmap_array

#     # Convert memmap to .mat file
#     print("Converting memmap to .mat file...")
#     data = np.memmap(mmap_filename, dtype='U2', mode='r', shape=(total_samples, num_snps))
#     io.savemat(output_file, {'combined_genotypes': data})

#     # Remove the temporary memmap file
#     os.remove(mmap_filename)

#     print(f"Combined data saved to {output_file}")
#     print(f"Shape of combined data: {(total_samples, num_snps)}")

# def main():
#     input_folder = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_seq'
#     output_file = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_combined_genotypes_2.mat'
#     chunk_size = 50 
#     combine_mat_files(input_folder, output_file, chunk_size)

# if __name__ == "__main__":
#     main()

