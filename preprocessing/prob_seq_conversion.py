#************************** Saving Files as .mat row wise SNPs for all chromosomes ********************#
import os
import gzip
import numpy as np
from scipy import io
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-reference_file', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can_weights_deepcombi_2.gen', help='Path to the folder containing .mat sample files')
parser.add_argument('-sample_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_weights_deepcombi_2', help='Path to the output folder')
parser.add_argument('-output_base_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_seq_weights_deepcombi_2', help='Path to the metadata file')

args = parser.parse_args()

def load_reference_data(reference_file):
    ref_data = {}
    chr_indices = {}
    with open(reference_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            chromosome = parts[0]
            if chromosome not in chr_indices:
                chr_indices[chromosome] = []
                ref_data[chromosome] = []
            chr_indices[chromosome].append(i)
            ref_data[chromosome].append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
    return ref_data, chr_indices

def process_sample_file(args):
    sample_file, ref_data, chr_indices, output_base_folder = args
    #print(f"Processing {sample_file}")
    
    base_name = os.path.basename(sample_file)
    
    results = []
    
    with gzip.open(sample_file, 'rt') as in_f:
        all_lines = in_f.readlines()
        
    for chromosome in sorted(chr_indices.keys()):
        genotypes = []
        for i in chr_indices[chromosome]:
            if i < len(all_lines):
                line = all_lines[i]
                probs = list(map(float, line.strip().split()))
                ref_allele, alt_allele = ref_data[chromosome][chr_indices[chromosome].index(i)]
                
                if max(probs) == probs[0]:  # Homozygous dominant
                    genotype = ref_allele + ref_allele
                elif max(probs) == probs[1]:  # Heterozygous
                    genotype = ref_allele + alt_allele
                else:  # Homozygous recessive
                    genotype = alt_allele + alt_allele
                
                genotypes.append(genotype)
        
        if genotypes:
            # Convert genotypes to a single row of space-separated genotypes
            genotypes_row = ' '.join(genotypes)
            
            # Create chromosome-specific output folder
            output_folder = os.path.join(output_base_folder, f"chromosome_{chromosome}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Save as .mat file
            output_file = os.path.join(output_folder, base_name.replace('.gen.gz', f'_chr{chromosome}.mat'))
            io.savemat(output_file, {'genotypes': genotypes_row})
            
            results.append(f"Processed and saved chromosome {chromosome} SNPs as .mat: {output_file}")
        else:
            results.append(f"No SNPs found for chromosome {chromosome} in {base_name}")
    
    return results

def main(reference_file, sample_folder, output_base_folder):
    
    # Load reference data
    print("Loading reference data and identifying SNPs for all chromosomes...")
    ref_data, chr_indices = load_reference_data(reference_file)
    
    # Prepare arguments for multiprocessing
    sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.gen.gz') and f.startswith('sample_')]
    args_list = [(os.path.join(sample_folder, f), ref_data, chr_indices, output_base_folder) for f in sample_files]
    
    # Use multiprocessing to process files
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        all_results = pool.map(process_sample_file, args_list)
    
    # Print results
    for results in all_results:
        for result in results:
            print(result)
    
    print("All files processed and saved as .mat files for each chromosome.")

if __name__ == "__main__":
    main(args.reference_file, args.sample_folder, args.output_base_folder)

#************************** Saving Files as .mat row wise SNPs for only one chromosome ********************#
# import os
# import gzip
# import numpy as np
# from scipy import io
# import multiprocessing as mp

# def load_reference_data(reference_file):
#     ref_data = []
#     chr1_indices = []
#     with open(reference_file, 'r') as f:
#         for i, line in enumerate(f):
#             parts = line.strip().split()
#             if parts[0] == '3':  # Check if chromosome is 1
#                 chr1_indices.append(i)
#                 ref_data.append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
#     #print(chr1_indices)
#     return ref_data, chr1_indices

# def process_sample_file(args):
#     sample_file, ref_data, chr1_indices, output_folder = args
#     print(sample_file)
    
#     base_name = os.path.basename(sample_file)
#     output_file = os.path.join(output_folder, base_name.replace('.gen.gz', '.mat'))
    
#     genotypes = []
#     with gzip.open(sample_file, 'rt') as in_f:
#         for i, line in enumerate(in_f):
#             if i in chr1_indices:
#                 probs = list(map(float, line.strip().split()))
#                 ref_allele, alt_allele = ref_data[chr1_indices.index(i)]
                
#                 if max(probs) == probs[0]:  # Homozygous dominant
#                     genotype = ref_allele + ref_allele
#                 elif max(probs) == probs[1]:  # Heterozygous
#                     genotype = ref_allele + alt_allele
#                 else:  # Homozygous recessive
#                     genotype = alt_allele + alt_allele
                
#                 genotypes.append(genotype)
    
#     # Convert genotypes to a single row of space-separated genotypes
#     genotypes_row = ' '.join(genotypes)
    
#     # Save as .mat file directly
#     io.savemat(output_file, {'genotypes': genotypes_row})
    
#     return f"Processed and saved chromosome 1 SNPs as .mat: {base_name}"

# def main():
#     reference_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen"
#     sample_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_seq/chr3"
    
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load reference data
#     print("Loading reference data and identifying chromosome 1 SNPs...")
#     ref_data, chr1_indices = load_reference_data(reference_file)
    
#     # Prepare arguments for multiprocessing
#     sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.gen.gz') and f.startswith('sample_')]
#     args_list = [(os.path.join(sample_folder, f), ref_data, chr1_indices, output_folder) for f in sample_files]
    
#     # Use multiprocessing to process files
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(process_sample_file, args_list)
    
#     # Print results
#     for result in results:
#         print(result)
    
#     print("All files processed and saved as .mat files with chromosome 1 SNPs only.")

# if __name__ == "__main__":
#     main()

#************************** Saving Files as .mat row wise SNPs for all chromosomes********************#
# import os
# import gzip
# import numpy as np
# from scipy import io
# import multiprocessing as mp

# def load_reference_data(reference_file):
#     ref_data = []
#     with open(reference_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             ref_data.append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
#     return ref_data

# def process_sample_file(args):
#     sample_file, ref_data, output_folder = args
#     base_name = os.path.basename(sample_file)
#     output_file = os.path.join(output_folder, base_name.replace('.gen.gz', '.mat'))
    
#     genotypes = []
#     with gzip.open(sample_file, 'rt') as in_f:
#         for i, line in enumerate(in_f):
#             probs = list(map(float, line.strip().split()))
#             ref_allele, alt_allele = ref_data[i]
            
#             if max(probs) == probs[0]:  # Homozygous dominant
#                 genotype = ref_allele + ref_allele
#             elif max(probs) == probs[1]:  # Heterozygous
#                 genotype = ref_allele + alt_allele
#             else:  # Homozygous recessive
#                 genotype = alt_allele + alt_allele
            
#             genotypes.append(genotype)
    
#     # Convert genotypes to a single row of space-separated genotypes
#     genotypes_row = ' '.join(genotypes)
    
#     # Save as .mat file directly
#     io.savemat(output_file, {'genotypes': genotypes_row})
    
#     return f"Processed and saved as .mat: {base_name}"

# def main():
#     reference_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen"
#     sample_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_seq"
    
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load reference data
#     print("Loading reference data...")
#     ref_data = load_reference_data(reference_file)
    
#     # Prepare arguments for multiprocessing
#     sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.gen.gz') and f.startswith('sample_')]
#     args_list = [(os.path.join(sample_folder, f), ref_data, output_folder) for f in sample_files]
    
#     # Use multiprocessing to process files
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(process_sample_file, args_list)
    
#     # Print results
#     for result in results:
#         print(result)
    
#     print("All files processed and saved as .mat files.")

# if __name__ == "__main__":
#     main()

# #************************** Saving Files as .mat row wise SNPs********************#
# import os
# import gzip
# import numpy as np
# from scipy import io

# def load_reference_data(reference_file):
#     ref_data = []
#     with open(reference_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             ref_data.append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
#     return ref_data

# def process_sample_file(sample_file, ref_data, output_folder):
#     base_name = os.path.basename(sample_file)
#     output_file = os.path.join(output_folder, base_name.replace('.gen.gz', '.mat'))
    
#     genotypes = []
#     with gzip.open(sample_file, 'rt') as in_f:
#         for i, line in enumerate(in_f):
#             probs = list(map(float, line.strip().split()))
#             ref_allele, alt_allele = ref_data[i]
            
#             if max(probs) == probs[0]:  # Homozygous dominant
#                 genotype = ref_allele + ref_allele
#             elif max(probs) == probs[1]:  # Heterozygous
#                 genotype = ref_allele + alt_allele
#             else:  # Homozygous recessive
#                 genotype = alt_allele + alt_allele
            
#             genotypes.append(genotype)
    
#     # Convert genotypes to a numpy array of characters
#     genotypes_array = np.array(genotypes, dtype='U2')
    
#     # Save as .mat file
#     io.savemat(output_file, {'genotypes': genotypes_array})
    
#     print(f"Processed and saved as .mat: {base_name}")

# def main():
#     reference_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen"
#     sample_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_seq"
    
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load reference data
#     print("Loading reference data...")
#     ref_data = load_reference_data(reference_file)
    
#     # Process each sample file
#     for filename in os.listdir(sample_folder):
#         if filename.endswith('.gen.gz') and filename.startswith('sample_'):
#             sample_file = os.path.join(sample_folder, filename)
#             process_sample_file(sample_file, ref_data, output_folder)
    
#     print("All files processed and saved as .mat files.")

# if __name__ == "__main__":
#     main()
    
#************************** Saving Files as .gen row wise SNPs********************#
# import os
# import gzip

# def load_reference_data(reference_file):
#     ref_data = []
#     with open(reference_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             ref_data.append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
#     return ref_data

# def process_sample_file(sample_file, ref_data, output_folder):
#     base_name = os.path.basename(sample_file)
#     output_file = os.path.join(output_folder, base_name[:-3])  # Remove .gz extension
    
#     with gzip.open(sample_file, 'rt') as in_f, open(output_file, 'w') as out_f:
#         for i, line in enumerate(in_f):
#             probs = list(map(float, line.strip().split()))
#             ref_allele, alt_allele = ref_data[i]
            
#             if max(probs) == probs[0]:  # Homozygous dominant
#                 genotype = ref_allele + ref_allele
#             elif max(probs) == probs[1]:  # Heterozygous
#                 genotype = ref_allele + alt_allele
#             else:  # Homozygous recessive
#                 genotype = alt_allele + alt_allele
            
#             out_f.write(genotype + '\n')
    
#     print(f"Processed: {base_name}")

# def main():
#     reference_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_brea_can.gen"
#     sample_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can"
#     output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_seq"
    
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load reference data
#     print("Loading reference data...")
#     ref_data = load_reference_data(reference_file)
    
#     # Process each sample file
#     for filename in os.listdir(sample_folder):
#         if filename.endswith('.gen.gz') and filename.startswith('sample_'):
#             sample_file = os.path.join(sample_folder, filename)
#             process_sample_file(sample_file, ref_data, output_folder)
    
#     print("All files processed.")

# if __name__ == "__main__":
#     main()

