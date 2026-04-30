#************************** Process the geno.mat for all samples at once for all chromosomes ***********************#
import os
import numpy as np
from scipy import io
import multiprocessing
from functools import partial
import traceback
import argparse

parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-input_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros_seq/pros_seq', help='Path to the folder containing .mat sample files')
parser.add_argument('-output_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros_seq/combined', help='Path to the output folder')
parser.add_argument('-num_processes', type=int, default=1, help='Maximum number of processes to use')
args = parser.parse_args()

def process_genetic_data(input_file_path, output_file_path):
    print(f"Loading data from: {input_file_path}")
    
    # Get file info without loading the entire file
    mat_info = io.whosmat(input_file_path)
    for var_name, shape, dtype in mat_info:
        if var_name == 'combined_genotypes':
            n_subjects, n_snps = shape
            break
    else:
        raise ValueError("'combined_genotypes' not found in the .mat file")
    
    print(f"Raw data shape: {(n_subjects, n_snps)}")
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of SNPs: {n_snps}")
    
    # Initialize output array
    output = np.memmap(output_file_path + '.tmp', dtype='U1', mode='w+', shape=(n_subjects, n_snps, 2))
    
    # Process data in chunks
    chunk_size = 1
    for start in range(0, n_subjects, chunk_size):
        end = min(start + chunk_size, n_subjects)
        print(f"Processing subjects {start} to {end}")
        
        # Load a chunk of data
        chunk_data = io.loadmat(input_file_path, variable_names=['combined_genotypes'])
        chunk = chunk_data['combined_genotypes'][start:end, :]
        
        for subject in range(chunk.shape[0]):
            for snp in range(n_snps):
                genotype = chunk[subject, snp]
                if isinstance(genotype, np.bytes_):
                    genotype = genotype.decode('utf-8')
                
                if len(genotype) == 1:
                    output[start + subject, snp, 0] = genotype
                    output[start + subject, snp, 1] = genotype
                elif len(genotype) == 2:
                    output[start + subject, snp, 0] = genotype[0]
                    output[start + subject, snp, 1] = genotype[1]
                else:
                    print(f"Warning: Unexpected genotype format at subject {start + subject}, SNP {snp}: {genotype}")
    
    print("Finished processing all subjects")
    
    # Save the converted data to a new .mat file
    print(f"Saving converted data to: {output_file_path}")
    io.savemat(output_file_path, {'genotype_data': output})
    
    # Remove the temporary memmap file
    os.remove(output_file_path + '.tmp')
    
    print(f"Converted data shape: {output.shape}")
    return output


def process_chromosome(input_folder, output_folder, chrom):
    input_file = os.path.join(input_folder, f'chr{chrom}.mat')
    output_file = os.path.join(output_folder, f'chr{chrom}_3d.mat')
    
    if os.path.exists(input_file):
        print(f"\nProcessing Chromosome {chrom}")
        try:
            genetic_data = process_genetic_data(input_file, output_file)
            print(f"\nChromosome {chrom} processing completed successfully")
            print(f"Sample data (first 5 SNPs for first subject):")
            print(genetic_data[0, :5, :])
            
            print("\nData statistics:")
            print(f"Total number of subjects: {genetic_data.shape[0]}")
            print(f"Total number of SNPs: {genetic_data.shape[1]}")
            unique_alleles = np.unique(genetic_data)
            print(f"Unique alleles in the data: {unique_alleles}")
            return f"Chromosome {chrom} processed successfully"
        except MemoryError:
            error_msg = f"MemoryError processing Chromosome {chrom}. Try reducing chunk_size or increasing system memory."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error processing Chromosome {chrom}: {str(e)}\n"
            error_msg += "Traceback:\n" + traceback.format_exc()
            print(error_msg)
            return error_msg
    else:
        return f"File for Chromosome {chrom} not found: {input_file}"

def process_all_chromosomes(input_folder, output_folder, num_processes=4):
    os.makedirs(output_folder, exist_ok=True)

    num_processes = min(num_processes if num_processes else multiprocessing.cpu_count() - 1, multiprocessing.cpu_count() - 1)

    print(f"Processing with {num_processes} processes")
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Prepare the partial function with fixed arguments
    process_func = partial(process_chromosome, input_folder, output_folder)
    
    # Process chromosomes in parallel
    results = pool.map(process_func, range(1, 23))
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    
    # Print results and collect errors
    errors = []
    for result in results:
        print(result)
        if result.startswith("Error"):
            errors.append(result)
    
    # Print summary of errors
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
    else:
        print("\nAll chromosomes processed successfully")

def main(input_folder,output_folder,num_processes=None):

    process_all_chromosomes(input_folder, output_folder, num_processes)
    print("\nProcessing of all chromosomes completed")

if __name__ == "__main__":
    main(args.input_folder,
         args.output_folder,
         args.num_processes)

#************************** Process the geno.mat for all samples at once ***********************#
# import numpy as np
# from scipy import io

# def process_genetic_data(input_file_path, output_file_path):
#     print(f"Loading data from: {input_file_path}")
#     # Load the .mat file
#     mat_data = io.loadmat(input_file_path)
#     raw_data = mat_data['combined_genotypes']
    
#     n_subjects, n_snps = raw_data.shape
    
#     print(f"Raw data shape: {raw_data.shape}")
#     print(f"Number of subjects: {n_subjects}")
#     print(f"Number of SNPs: {n_snps}")
    
#     # Initialize the output array
#     output = np.zeros((n_subjects, n_snps, 2), dtype='U1')
#     print(f"Initialized output array with shape: {output.shape}")
    
#     # Counter for progress updates
#     update_interval = max(1, n_subjects // 10)
    
#     for subject in range(n_subjects):
#         if subject % update_interval == 0:
#             print(f"Processing subject {subject}/{n_subjects}")
        
#         for snp in range(n_snps):
#             # Extract the single-character genotype
#             genotype = raw_data[subject, snp]
            
#             # Convert from numpy.bytes_ to string if necessary
#             if isinstance(genotype, np.bytes_):
#                 genotype = genotype.decode('utf-8')
            
#             # Assign to output array
#             # If it's a single character, duplicate it for both alleles
#             if len(genotype) == 1:
#                 output[subject, snp, 0] = genotype
#                 output[subject, snp, 1] = genotype
#             elif len(genotype) == 2:
#                 output[subject, snp, 0] = genotype[0]
#                 output[subject, snp, 1] = genotype[1]
#             else:
#                 print(f"Warning: Unexpected genotype format at subject {subject}, SNP {snp}: {genotype}")
#                 raise ValueError(f"Unexpected genotype format at subject {subject}, SNP {snp}: {genotype}")
    
#     print("Finished processing all subjects")
    
#     # Check if the conversion was successful
#     expected_shape = (n_subjects, n_snps, 2)
#     if output.shape != expected_shape:
#         print(f"Error: Conversion resulted in unexpected shape")
#         raise ValueError(f"Conversion error: Expected shape {expected_shape}, but got {output.shape}")
#     else:
#         print("Conversion successful: Output shape matches expected shape")
    
#     print(f"Saving converted data to: {output_file_path}")
#     # Save the converted data to a new .mat file
#     io.savemat(output_file_path, {'genotype_data': output})
    
#     print(f"Converted data shape: {output.shape}")
#     return output

# def main():
#     input_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_seq/combined/brea_chr1.mat'
#     output_file_path = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_seq/combined/brea_chr1_3d.mat'

#     try:
#         genetic_data = process_genetic_data(input_file_path, output_file_path)
#         print("\nProcessing completed successfully")
#         print(f"\nSample data (first 5 SNPs for first subject):")
#         print(genetic_data[0, :5, :])
        
#         print("\nData statistics:")
#         print(f"Total number of subjects: {genetic_data.shape[0]}")
#         print(f"Total number of SNPs: {genetic_data.shape[1]}")
#         unique_alleles = np.unique(genetic_data)
#         print(f"Unique alleles in the data: {unique_alleles}")
        
#     except ValueError as e:
#         print(f"Error occurred during processing: {e}")
#     except Exception as e:
#         print(f"Unexpected error occurred: {e}")

# if __name__ == "__main__":
#     main()
