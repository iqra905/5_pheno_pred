# **************************** Version 3 - works with the chr1 SNPs ***************#
import numpy as np
from scipy import io
import h5py

def view_mat_file(file_path):
    print(f"Viewing contents of: {file_path}")

    try:
        # Try loading with scipy.io.loadmat first
        try:
            mat_contents = io.loadmat(file_path)
            print("File loaded successfully with scipy.io.loadmat")
        except NotImplementedError:
            # If that fails, try using h5py
            print("scipy.io.loadmat failed. Trying with h5py...")
            with h5py.File(file_path, 'r') as file:
                mat_contents = {key: file[key][:] for key in file.keys()}
            print("File loaded successfully with h5py")

        # Display information about the contents
        print("\nFile contents:")
        for key, value in mat_contents.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: numpy array of shape {value.shape} and dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

        # Display a sample of the data
        main_key = 'combined_genotypes'  # Assuming this is the main data key - combined
        #main_key = 'genotype_data'  # Assuming this is the main data key - combined_3d
        if main_key in mat_contents:
            data = mat_contents[main_key]
            print(f"\nSample of {main_key} (first 5 rows, first 10 columns):")
            print(data[:5, :10])

            # Display some statistics
            if np.issubdtype(data.dtype, np.number):
                print(f"\nStatistics of {main_key}:")
                print(f"  Mean: {np.mean(data)}")
                print(f"  Std Dev: {np.std(data)}")
                print(f"  Min: {np.min(data)}")
                print(f"  Max: {np.max(data)}")
            else:
                print(f"\nUnique values in {main_key}: {np.unique(data)}")
        else:
            print(f"\nWarning: '{main_key}' not found in the file.")

    except Exception as e:
        print(f"Error loading or processing the file: {e}")

if __name__ == "__main__":
    #file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_genotypes.mat'
    #file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_seq/combined/pros_can_chr22_3d.mat'
    file_path = '/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise_pruned/brea_seq/0.05/combined/chr6.mat'
    #file_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/preprocessing/chr1.mat'


    #/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_seq/chr3
    view_mat_file(file_path)
    
# **************************** Version 2 - works with the chr1 SNPs ***************#
# import numpy as np
# from scipy import io
# import os
# import sys
# from collections import Counter

# def view_mat_file(file_path, chunk_size=1000, num_random_samples=5):
#     if not os.path.exists(file_path):
#         print(f"Error: File {file_path} does not exist.")
#         return

#     try:
#         # Get information about the .mat file
#         mat_contents = io.whosmat(file_path)
#         print(mat_contents)
#         if not mat_contents:
#             print(f"Error: No variables found in {file_path}")
#             return

#         var_name, shape, data_type = mat_contents[0]
#         total_samples, num_snps = shape

#         print(f"File: {file_path}")
#         print(f"Variable name: {var_name}")
#         print(f"Shape of combined data: {shape}")
#         print(f"Number of samples: {total_samples}")
#         print(f"Number of SNPs: {num_snps}")
#         print(f"Data type: {data_type}")
        
#         # Calculate file size
#         file_size = os.path.getsize(file_path)
#         print(f"File size: {file_size / (1024 * 1024):.2f} MB")

#         # Function to safely convert byte string to regular string
#         def byte_to_str(b):
#             return b.decode('utf-8') if isinstance(b, bytes) else str(b)

#         # Read and print first few SNPs of the first sample
#         first_sample = io.loadmat(file_path, variable_names=[var_name], 
#                                   squeeze_me=True, struct_as_record=False,
#                                   chars_as_strings=True,
#                                   mat_dtype=True)
#         first_genotypes = first_sample[var_name][0]
#         print("\nFirst few SNPs of first sample:")
#         print(' '.join(byte_to_str(g) for g in first_genotypes[:10]))

#         # Read and print first few SNPs of the last sample
#         last_sample = io.loadmat(file_path, variable_names=[var_name], 
#                                  squeeze_me=True, struct_as_record=False,
#                                  chars_as_strings=True,
#                                  mat_dtype=True)
#         last_genotypes = last_sample[var_name][-1]
#         print("\nFirst few SNPs of last sample:")
#         print(' '.join(byte_to_str(g) for g in last_genotypes[:10]))

#         # Print random samples
#         print(f"\nRandom samples (first few SNPs of {num_random_samples} random samples):")
#         for _ in range(num_random_samples):
#             random_index = np.random.randint(0, total_samples)
#             chunk_start = (random_index // chunk_size) * chunk_size
#             chunk = io.loadmat(file_path, variable_names=[var_name], 
#                                squeeze_me=True, struct_as_record=False,
#                                chars_as_strings=True,
#                                mat_dtype=True)
#             random_genotypes = chunk[var_name][random_index]
#             print(f"Sample {random_index}: {' '.join(byte_to_str(g) for g in random_genotypes[:10])}")

#         # Calculate and print some statistics
#         print("\nCalculating statistics...")
#         genotype_counter = Counter()

#         for start_idx in range(0, total_samples, chunk_size):
#             end_idx = min(start_idx + chunk_size, total_samples)
#             chunk = io.loadmat(file_path, variable_names=[var_name], 
#                                squeeze_me=True, struct_as_record=False,
#                                chars_as_strings=True,
#                                mat_dtype=True)
#             chunk_data = chunk[var_name][start_idx:end_idx]
            
#             for sample in chunk_data:
#                 genotype_counter.update(map(byte_to_str, sample))
            
#             print(f"Processed {end_idx}/{total_samples} samples")

#         unique_genotypes = list(genotype_counter.keys())
#         print(f"\nNumber of unique genotypes: {len(unique_genotypes)}")
#         print("Top 5 most common genotypes:")
#         for genotype, count in genotype_counter.most_common(5):
#             print(f"{genotype}: {count}")

#     except Exception as e:
#         print(f"An error occurred while reading the file: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         mat_file_path = sys.argv[1]
#     else:
#         mat_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_combined_genotypes_2.mat'
    
#     view_mat_file(mat_file_path)
    

# from scipy import io
# import numpy as np

# def view_mat_file(file_path, chunk_size=1000):
#     try:
#         mat_contents = io.loadmat(file_path)
        
#         if 'combined_genotypes' in mat_contents:
#             combined_genotypes = mat_contents['combined_genotypes']
#             print(f"Shape of combined data: {combined_genotypes.shape}")
#             print(f"Number of samples: {combined_genotypes.shape[0]}")
#             print(f"Number of SNPs: {combined_genotypes.shape[1]}")
            
#             print("\nFirst few genotypes of first sample:")
#             print(' '.join(combined_genotypes[0, :10].flatten()))
            
#             print("\nFirst few genotypes of last sample:")
#             print(' '.join(combined_genotypes[-1, :10].flatten()))
            
#             # Check a few random samples
#             for i in range(3):
#                 random_index = np.random.randint(0, combined_genotypes.shape[0])
#                 print(f"\nRandom sample (index {random_index}):")
#                 print(' '.join(combined_genotypes[random_index, :10].flatten()))
#         else:
#             print("No 'combined_genotypes' variable found in the file.")
    
#     except Exception as e:
#         print(f"Error reading the MAT file: {e}")

# if __name__ == "__main__":
#     mat_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_combined_genotypes_2.mat'
#     view_mat_file(mat_file_path)


# from scipy import io
# import numpy as np

# def view_mat_file(file_path):
#     mat_contents = io.loadmat(file_path)
    
#     print(f"Contents of {file_path}:")
    
#     if 'combined_genotypes' in mat_contents:
#         combined_genotypes = mat_contents['combined_genotypes']
#         print(f"Shape of combined data: {combined_genotypes.shape}")
#         print(f"Number of samples: {combined_genotypes.shape[0]}")
#         print(f"Number of SNPs: {combined_genotypes.shape[1]}")
#         print("\nFirst few genotypes of first sample:")
#         print(' '.join(combined_genotypes[0, :10]))
#         print("\nFirst few genotypes of last sample:")
#         print(' '.join(combined_genotypes[-1, :10]))
#     else:
#         print("Unexpected data format")

# if __name__ == "__main__":
#     mat_file_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_geno.mat'
#     view_mat_file(mat_file_path)
    

#********************* Version to view sample.mat files ********************#
# import sys
# from scipy import io
# import numpy as np

# # def view_mat_file(file_path):
# #     # Load the .mat file
# #     mat_contents = io.loadmat(file_path)
    
# #     print(f"Contents of {file_path}:")
    
# #     # The data is stored directly, so we access it with an empty string key
# #     if 'genotypes' in mat_contents:
# #         genotypes = mat_contents['genotypes'][0].split()
# #         print(f"Number of genotypes: {len(genotypes)}")
# #         print(f"First few genotypes: {' '.join(genotypes[:5])}")
# #         print(f"Last few genotypes: {' '.join(genotypes[-5:])}")
# #     else:
# #         print("Unexpected data format")

# def view_mat_file(file_path):
#     # Load the .mat file
#     mat_contents = io.loadmat(file_path)
    
#     print(f"Contents of {file_path}:")
#     # Print the keys (variable names) in the .mat file
#     print("Variables in the .mat file:")
#     for key in mat_contents.keys():
#         if not key.startswith('__'):  # Skip metadata
#             print(f"Variable: {key}")
            
#             # Print information about the variable
#             var = mat_contents[key]
#             print(f"  Shape: {var.shape}")
#             print(f"  Type: {var.dtype}")

#             if isinstance(var, np.ndarray) and var.dtype.kind in ['U', 'S']:
#                 genotypes = var[0].split()
#                 print(f"Number of genotypes: {len(genotypes)}")
#                 print(f"First few genotypes: {' '.join(genotypes[:5])}")
#                 print(f"Last few genotypes: {' '.join(genotypes[-5:])}")
#             else:
#                 print("Unexpected data format")
#             print()
            
#             # # If it's a small array, print its contents
#             # if var.size < 10:
#             #     print(f"  Contents: {var}")
#             # else:
#             #     print(f"  First few elements: {var.flatten()[:5]}")
#             # print()

# if __name__ == "__main__":
#     # if len(sys.argv) < 2:
#     #     print("Usage: python view_mat.py <path_to_mat_file>")
#     # else:
#     #     view_mat_file(sys.argv[1])
#     # Specify the path to your .mat file here
#     mat_file_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_seq/sample_36818.mat'
    
#     view_mat_file(mat_file_path)