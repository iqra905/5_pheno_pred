import pandas as pd
import numpy as np
import h5py
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from multiprocessing import Pool, cpu_count
import os
import glob
import time
import re

parser = argparse.ArgumentParser(description='Statistical Analysis for GWAS')
parser.add_argument('-genotype_dir', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/gen_data_5M_H5_files/col', help='Directory containing chromosome-wise genotype data files')
parser.add_argument('-phenotype_path', type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno_updated/col_can.xlsx', help='Path to the phenotype data file')
parser.add_argument('-phenotype_column', type=str, default='crc', help='Name of the phenotype column to analyze')
parser.add_argument('-output_dir', type=str, default='/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new_study_split/col', help='Directory for saving output files')
parser.add_argument('-use_pcs', action='store_true', help='Include principal components as covariates')
parser.add_argument('-use_age', action='store_true', help='Include age as a covariate')
parser.add_argument('-use_gender', action='store_true', help='Include gender as a covariate')
parser.add_argument('-num_processes', type=int, default=30, help='Number of processes for parallel computation')
parser.add_argument('-p_value_threshold', type=float, default=0.05, help='P-value threshold for significance')
parser.add_argument('-train_split', type=float, default=0.8, help='Percentage of samples to use for training (default: 0.8)')
parser.add_argument('-random_seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-chromosomes', type=str, default='all', help='Chromosomes to analyze: "all" for all chromosomes, comma-separated numbers (e.g., "1,3,5"), range (e.g., "1-5"). Default: all')

args = parser.parse_args()

def create_random_train_test_split(phenotype_path, train_percentage, random_seed, output_dir):
    # Load phenotype data
    phenotype_data = pd.read_excel(phenotype_path)
    total_samples = len(phenotype_data)
    
    # Create random split indices
    if train_percentage == 1.0:
        train_indices = np.arange(total_samples)
        test_indices = np.array([], dtype=int)  # Empty array for test indices
        
        if output_dir:
            np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
            np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
            
        print(f"Using all {total_samples} samples for training, 0 samples for testing")
        return train_indices, test_indices
    else:
        indices = np.arange(total_samples)
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_percentage,
            random_state=random_seed,
            shuffle=True
        )
    
        # Sort indices for better data access patterns
        train_indices.sort()
        test_indices.sort()
    
        # Save indices
        if output_dir:
            np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
            np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    
        print(f"Random split indices saved: {len(train_indices)} training samples, {len(test_indices)} test samples")
        return train_indices, test_indices
    raise

def load_phenotype_data(file_path, phenotype_column, indices, use_pcs=True, use_age=False, use_gender=False):
    """Load phenotype data and selected covariates, using only specified indices."""
    try:
        phenotype_data = pd.read_excel(file_path)
        
        # Initialize list of columns with phenotype
        selected_columns = [phenotype_column]
        
        # Add covariates based on arguments
        if use_pcs:
            pc_columns = [f'PC{i}' for i in range(1, 11)]
            selected_columns.extend(pc_columns)
        
        if use_age:
            selected_columns.append('Agexit')
        
        if use_gender:
            selected_columns.append('Sex')
        
        # Verify all columns exist
        missing_columns = [col for col in selected_columns if col not in phenotype_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in phenotype data: {missing_columns}")
        
        # Select data using indices
        if indices is None:  # Use all data
            phenotype_data = phenotype_data[selected_columns]
        else:
            phenotype_data = phenotype_data[selected_columns].iloc[indices]
        
        print(f"Phenotype data loaded successfully. Shape: {phenotype_data.shape}")
        print(f"Using columns: {', '.join(selected_columns)}")
        return phenotype_data
    except Exception as e:
        print(f"Error loading phenotype data: {e}")
        raise

def perform_logistic_regression(args):
    """Perform logistic regression for a single SNP."""
    snp_data, phenotype_data, snp_index, phenotype_column = args
    try:
        # Use the already extracted SNP data (no indexing needed)
        X_snp = snp_data.reshape(-1, 1)
        X_covariates = phenotype_data.drop(columns=[phenotype_column])
        
        if not X_covariates.empty:
            scaler = StandardScaler()
            X_covariates_scaled = scaler.fit_transform(X_covariates)
            X = np.concatenate((X_snp, X_covariates_scaled), axis=1)
        else:
            X = X_snp
            
        y = phenotype_data[phenotype_column]
        X = sm.add_constant(X)
        
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        snp_p_value = result.pvalues.iloc[1]
        return snp_index, snp_p_value

    except Exception as e:
        print(f"Error performing logistic regression for SNP {snp_index}: {e}")
        # Use 1.0 (no significance) as placeholder instead of None
        return snp_index, 1.0

def process_chromosome_parallel_snps(chr_file, phenotype_data, phenotype_column, p_value_threshold, output_dir, indices, num_processes):
    """Process a single chromosome file with parallel SNP analysis."""
    chr_num = os.path.basename(chr_file).split('_')[-1].split('.')[0][3:]
    print(f"\nProcessing chromosome {chr_num}")
    start_time = time.time()
    
    # Get information about the chromosome file without loading all data
    with h5py.File(chr_file, 'r') as f:
        total_snps = f['data'].shape[1]
    
    print(f"Chromosome {chr_num} has {total_snps} SNPs")
    
    # Process SNPs in batches to minimize memory usage
    batch_size = 50000  # Adjust based on available memory
    all_snp_results = []
    failed_snps_count = 0
    processed_snps = 0
    
    for batch_start in range(0, total_snps, batch_size):
        batch_end = min(batch_start + batch_size, total_snps)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch of SNPs {batch_start}-{batch_end-1} ({current_batch_size} SNPs)")
        
        # Load only the current batch of SNPs
        with h5py.File(chr_file, 'r') as f:
            if indices is None:
                batch_data = f['data'][:, batch_start:batch_end]
            else:
                batch_data = f['data'][indices, batch_start:batch_end]
        
        # Prepare arguments for parallel processing - pass only the specific SNP data
        process_args = []
        for i in range(current_batch_size):
            snp_index = batch_start + i
            snp_data = batch_data[:, i]  # Extract just this SNP's data
            process_args.append((snp_data, phenotype_data, snp_index, phenotype_column))
        
        # Free the batch data to save memory
        del batch_data
        
        # Process SNPs in parallel
        with Pool(processes=num_processes) as pool:
            # Use dynamic chunk size for better load balancing
            chunk_size = max(1, current_batch_size // (num_processes * 5))
            results_iter = pool.imap_unordered(perform_logistic_regression, process_args, chunksize=chunk_size)
            
            # Process results as they come in
            batch_results = []
            for result in results_iter:
                snp_index, p_value = result
                batch_results.append((snp_index, p_value))
                
                if p_value == 1.0:
                    failed_snps_count += 1
            
            # Add batch results to overall results
            all_snp_results.extend(batch_results)
        
        # Update progress
        processed_snps += current_batch_size
        print(f"Chromosome {chr_num}: Processed {processed_snps}/{total_snps} SNPs")
        
        # Free memory
        del process_args
    
    # Create and save results dataframe
    results_df = pd.DataFrame({
        'Chromosome': chr_num,
        'SNP_Index': [result[0] for result in all_snp_results],
        'P_Value': [result[1] for result in all_snp_results],
        'Is_Significant': [p_value <= p_value_threshold and p_value != 1.0 for _, p_value in all_snp_results],
        'Failed_Analysis': [p_value == 1.0 for _, p_value in all_snp_results]
    })
    
    # Sort results by SNP index to ensure they're in the correct order
    results_df = results_df.sort_values('SNP_Index')
    
    # Save all results
    all_results_file = os.path.join(output_dir, f'train_set_all_snps_chr{chr_num}.csv')
    results_df.to_csv(all_results_file, index=False)
    
    # Save significant results separately
    significant_df = results_df[results_df['Is_Significant']]
    if not significant_df.empty:
        significant_file = os.path.join(output_dir, f'train_set_significant_snps_chr{chr_num}.csv')
        significant_df.to_csv(significant_file, index=False)
        print(f"Saved {len(significant_df)} significant SNPs for chromosome {chr_num}")
    
    # Report failed SNPs
    if failed_snps_count > 0:
        failed_df = results_df[results_df['Failed_Analysis']]
        failed_file = os.path.join(output_dir, f'train_set_failed_snps_chr{chr_num}.csv')
        failed_df.to_csv(failed_file, index=False)
        print(f"Warning: {failed_snps_count} SNPs failed analysis on chromosome {chr_num} (assigned p-value=1.0)")
    
    # Report performance
    duration = time.time() - start_time
    print(f"Completed chromosome {chr_num} in {duration:.2f} seconds")
    
    return len(significant_df)

def filter_chromosome_files(chr_files, chromosome_selection):
    if chromosome_selection.lower() == 'all':
        return chr_files
    
    # Function to extract chromosome number/name from filename
    def extract_chr_info(filename):
        # Extract the chromosome name (e.g., 'chr1', 'chrX')
        chr_part = os.path.basename(filename).split('_')[-1].split('.')[0]
        return chr_part[3:]  # Remove 'chr' prefix
    
    # Map of all available chromosome files
    chr_map = {extract_chr_info(f): f for f in chr_files}
    
    selected_chrs = []
    
    # Check if it's a range (e.g., '1-5')
    if '-' in chromosome_selection:
        try:
            start, end = chromosome_selection.split('-')
            start, end = int(start), int(end)
            for i in range(start, end + 1):
                selected_chrs.append(str(i))
        except ValueError:
            print(f"Warning: Invalid range format '{chromosome_selection}'. Using all chromosomes.")
            return chr_files
    
    # Otherwise, treat as comma-separated list
    else:
        selected_chrs = [c.strip() for c in chromosome_selection.split(',')]
    
    # Filter files based on selection
    filtered_files = []
    for chr_id in selected_chrs:
        if chr_id in chr_map:
            filtered_files.append(chr_map[chr_id])
        else:
            print(f"Warning: Chromosome {chr_id} not found in data directory")
    
    if not filtered_files:
        print(f"No matching chromosome files found for selection: {chromosome_selection}")
        print(f"Available chromosomes: {', '.join(sorted(chr_map.keys()))}")
        return []
    
    return filtered_files

def process_chromosomes_sequential(chr_files, phenotype_data, phenotype_column, p_value_threshold, num_processes, output_dir, indices):
    """Process chromosome files sequentially, with parallel SNP analysis within each chromosome."""
    total_significant_snps = 0
    total_start_time = time.time()
    
    print(f"Processing {len(chr_files)} chromosomes sequentially")
    
    for chr_idx, chr_file in enumerate(chr_files):
        print(f"Starting chromosome {chr_idx+1}/{len(chr_files)}")
        significant_snps = process_chromosome_parallel_snps(
            chr_file, phenotype_data, phenotype_column, p_value_threshold, 
            output_dir, indices, num_processes
        )
        total_significant_snps += significant_snps
        
    total_duration = time.time() - total_start_time
    print(f"Completed all chromosomes in {total_duration:.2f} seconds")
    return total_significant_snps

if __name__ == "__main__":
    try:
        print("Starting GWAS analysis with train-test split...")
        print(f"Using sequential chromosome processing with parallel SNP analysis")
        start_time = time.time()

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Results will be saved in: {args.output_dir}")
        
        # Create and save random train/test split
        train_indices, test_indices = create_random_train_test_split(
            args.phenotype_path,
            args.train_split,
            args.random_seed,
            args.output_dir
        )
        
        print(f"Total number of training samples: {len(train_indices)}")
        print(f"Total number of test samples: {len(test_indices)}")
        print(f"Total number of samples: {len(train_indices) + len(test_indices)}")
        print(f"Using {len(train_indices)} train samples for analysis")
        #print(f"Using {len(test_indices)} test samples for analysis")

        # Load phenotype data with indices 
        phenotype_data = load_phenotype_data(
            args.phenotype_path,
            args.phenotype_column,
            train_indices, # Use test_indices instead of train_indices
            args.use_pcs,
            args.use_age,
            args.use_gender
        )

        # Get chromosome files and sort them numerically
        all_chr_files = glob.glob(os.path.join(args.genotype_dir, '*chr*.h5'))
        
        # Extract chromosome number and sort numerically
        def get_chr_number(filename):
            # Extract the chromosome number from filename (e.g., 'chr1.h5' → 1)
            chr_name = os.path.basename(filename).split('_')[-1].split('.')[0]
            # Remove 'chr' prefix and convert to integer
            try:
                return int(chr_name[3:])
            except ValueError:
                # Handle special cases like 'chrX', 'chrY'
                special_chr = {'X': 23, 'Y': 24, 'M': 25}
                return special_chr.get(chr_name[3:], 999)  # Default high number for unknown
        
        # Sort chromosomes numerically (1,2,3...10,11 instead of 1,10,11,2...)
        all_chr_files = sorted(all_chr_files, key=get_chr_number)
        
        if not all_chr_files:
            raise ValueError(f"No chromosome files found with pattern: chr*.h5 in directory: {args.genotype_dir}")
        
        # Filter chromosome files based on user selection
        print(f"Chromosome selection: {args.chromosomes}")
        chr_files = filter_chromosome_files(all_chr_files, args.chromosomes)
        
        if not chr_files:
            raise ValueError(f"No chromosome files selected for analysis. Check your '-chromosomes' parameter.")
        
        print(f"Found {len(all_chr_files)} total chromosome files")
        print(f"Selected {len(chr_files)} chromosome files for analysis")
        print(f"Processing chromosomes: {[os.path.basename(f).split('_')[-1].split('.')[0][3:] for f in chr_files]}")
        
        # Process chromosomes sequentially with parallel SNP analysis
        total_significant_snps = process_chromosomes_sequential(
            chr_files, 
            phenotype_data,
            args.phenotype_column,
            args.p_value_threshold,
            args.num_processes, 
            args.output_dir,
            train_indices  # Use test_indices instead of train_indices
        )
        
        print(f"\nTotal significant SNPs across selected chromosomes: {total_significant_snps}")
        print(f"Analysis completed in {(time.time() - start_time):.2f} seconds")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise