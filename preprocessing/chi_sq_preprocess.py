import os
import gzip
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import re

def convert_probabilities_to_genotype(probs):
    return np.argmax(probs)

def perform_statistical_test(genotypes, labels, snp_index):
    case_genotypes = genotypes[labels == 1]
    control_genotypes = genotypes[labels == 0]
    
    observed = np.array([
        np.bincount(case_genotypes, minlength=3),
        np.bincount(control_genotypes, minlength=3)
    ])
    
    print(f"SNP {snp_index}: Observed frequencies:")
    print(observed)
    
    # Always attempt chi-square test first, with continuity correction
    try:
        print(f"SNP {snp_index}: Attempting Chi-square test with continuity correction")
        _, p_value, _, expected = chi2_contingency(observed, correction=True)
        print(f"SNP {snp_index}: Expected frequencies:")
        print(expected)
        print(f"SNP {snp_index}: Chi-square test successful")
    except ValueError as e:
        print(f"SNP {snp_index}: Chi-square test failed, collapsing table for Fisher's exact test. Error: {str(e)}")
        # If chi-square test fails, collapse the table and use Fisher's exact test
        collapsed_observed = np.array([
            [observed[0, 0], observed[0, 1] + observed[0, 2]],
            [observed[1, 0], observed[1, 1] + observed[1, 2]]
        ])
        _, p_value = fisher_exact(collapsed_observed)
        print(f"SNP {snp_index}: Collapsed table for Fisher's exact test:")
        print(collapsed_observed)
    
    print(f"SNP {snp_index}: p-value = {p_value}")
    return p_value

def process_snp_chunk(snp_indices, sample_files, input_folder, labels, p_value_threshold):
    significant_snps = []
    significant_genotypes = []
    
    chunk_start = min(snp_indices)
    chunk_end = max(snp_indices)
    chunk_size = len(snp_indices)
    
    print(f"\nProcessing SNP chunk: {chunk_start} to {chunk_end} (size: {chunk_size})")
    
    for snp_index in snp_indices:
        genotypes = []
        probabilities = []
        for sample_file in sample_files:
            with gzip.open(os.path.join(input_folder, sample_file), 'rt') as f:
                for i, line in enumerate(f):
                    if i == snp_index:
                        probs = line.strip().split()
                        genotypes.append(convert_probabilities_to_genotype(list(map(float, probs))))
                        probabilities.append(probs)
                        break
        
        try:
            p_value = perform_statistical_test(np.array(genotypes), labels, snp_index)
            if p_value < p_value_threshold:
                significant_snps.append(snp_index)
                significant_genotypes.append(probabilities)
                print(f"SNP {snp_index}: Significant (p-value = {p_value})")
            else:
                print(f"SNP {snp_index}: Not significant (p-value = {p_value})")
        except Exception as e:
            print(f"Error processing SNP {snp_index}: {str(e)}")
    
    print(f"Chunk {chunk_start}-{chunk_end} complete. Found {len(significant_snps)} significant SNPs.")
    return significant_snps, significant_genotypes

def get_sample_number(filename):
    return int(re.search(r'sample_(\d+)', filename).group(1))

def process_files(input_folder, output_folder, phenotype_file, first_5_columns_file, p_value_threshold, chunk_size=1000):
    start_time = time.time()
    print(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Read phenotype data
    print(f"\nStep 1: Reading phenotype data from {phenotype_file}")
    phenotype_data = pd.read_excel(phenotype_file)
    labels = phenotype_data['pros01'].values
    print(f"Phenotype data read. Found {len(labels)} samples.")

    # Create output folder if it doesn't exist
    print(f"\nStep 2: Creating output folder {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    print("Output folder created or already exists.")

    # Process first 5 columns file
    print(f"\nStep 3: Reading first 5 columns file {first_5_columns_file}")
    with gzip.open(first_5_columns_file, 'rt') as f_in:
        first_5_columns = [line.strip().split() for line in f_in]
    print(f"First 5 columns file read. Found {len(first_5_columns)} rows.")

    # Get list of sample files
    print(f"\nStep 4: Getting list of sample files from {input_folder}")
    sample_files = [f for f in os.listdir(input_folder) if f.endswith('.gen.gz') and f.startswith('sample_')]
    sample_files.sort(key=get_sample_number)
    print(f"Found {len(sample_files)} sample files.")
    print(f"Sample range: {sample_files[0]} to {sample_files[-1]}")

    # Get number of SNPs
    print("\nStep 5: Determining number of SNPs")
    num_snps = sum(1 for _ in gzip.open(os.path.join(input_folder, sample_files[0]), 'rt'))
    print(f"Processing {num_snps} SNPs for each of the {len(sample_files)} samples")

    # Process SNPs in chunks
    print(f"\nStep 6: Processing SNPs in chunks and performing statistical tests (p-value threshold: {p_value_threshold})")
    pool = mp.Pool(processes=mp.cpu_count())
    print(f"Using {mp.cpu_count()} CPU cores for processing")
    
    process_chunk = partial(process_snp_chunk, sample_files=sample_files, input_folder=input_folder, 
                            labels=labels, p_value_threshold=p_value_threshold)
    
    significant_snps = []
    significant_genotypes = []
    
    chunk_indices = [range(i, min(i + chunk_size, num_snps)) for i in range(0, num_snps, chunk_size)]
    
    for i, result in enumerate(tqdm(pool.imap(process_chunk, chunk_indices), total=len(chunk_indices), desc="Processing SNP chunks")):
        chunk_significant_snps, chunk_significant_genotypes = result
        significant_snps.extend(chunk_significant_snps)
        significant_genotypes.extend(chunk_significant_genotypes)
        print(f"Completed chunk {i+1}/{len(chunk_indices)}. Total significant SNPs so far: {len(significant_snps)}")
    
    pool.close()
    pool.join()
    
    print(f"Statistical tests complete. Found {len(significant_snps)} significant SNPs in total.")

    # Write significant SNPs to output files
    print(f"\nStep 7: Writing output files to {output_folder}")
    print("Writing first 5 columns for significant SNPs...")
    with gzip.open(os.path.join(output_folder, 'first_5_columns_chi_sq.gen.gz'), 'wt') as f_out:
        for idx in significant_snps:
            f_out.write('\t'.join(first_5_columns[idx]) + '\n')
    print("First 5 columns file written successfully.")

    print("Writing filtered sample files...")
    for i, sample_file in enumerate(tqdm(sample_files, desc="Writing sample files")):
        with gzip.open(os.path.join(output_folder, sample_file.replace('.gez.gz', '_chi_sq.gez.gz')), 'wt') as f_out:
            for snp_probabilities in significant_genotypes:
                probs = snp_probabilities[i]
                f_out.write(' '.join(probs) + '\n')
    print("All sample files written successfully.")

    end_time = time.time()
    print(f"\nProcessing complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    input_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can"
    output_folder = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_chi_sq"
    phenotype_file = "/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx"
    first_5_columns_file = "/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can.gen.gz"
    p_value_threshold = 0.05 
    chunk_size = 5000  
    process_files(input_folder, output_folder, phenotype_file, first_5_columns_file, p_value_threshold, chunk_size)