#!/usr/bin/env python3
"""
SNP Filtering Preprocessing Script

This script performs one-time filtering of chromosome-wise genotype files
based on a master SNP file, and saves the filtered variant information
for fast direct indexing during training.

Usage:
    python preprocess_snp_filtering.py

Output:
    Creates chr{N}_variants_filtered.gen files with filtered SNP indices
"""

import os
import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path

# Configuration
BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8"
MASTER_SNP_FILE = "/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen"


def load_master_snp_file(master_file_path):
    """
    Load the master 5M SNP filter file
    
    Returns:
        DataFrame with columns: chr, rsid, bp, ref_allele, alt_allele
    """
    print(f"Loading master SNP file from: {master_file_path}")
    start_time = time.time()
    
    # Check if file has header
    with open(master_file_path, 'r') as f:
        first_line = f.readline().strip()
        has_header = any(keyword in first_line.lower() 
                        for keyword in ['rsid', 'position', 'chr', 'ref', 'alt', 'allele'])
    
    # Load the master file
    master_snps = pd.read_csv(
        master_file_path,
        sep=r'\s+',
        header=0 if has_header else None,
        names=None if has_header else ['chr', 'rsid', 'bp', 'ref_allele', 'alt_allele'],
        dtype={'chr': str, 'rsid': str, 'ref_allele': str, 'alt_allele': str}
    )
    
    # Standardize column names
    if has_header:
        column_mapping = {}
        for col in master_snps.columns:
            col_lower = col.lower()
            if col_lower == 'chr' or col_lower == 'chromosome':
                column_mapping[col] = 'chr'
            elif 'rsid' in col_lower or col_lower == 'snp' or col_lower == 'id':
                column_mapping[col] = 'rsid'
            elif 'position' in col_lower or col_lower == 'bp' or col_lower == 'pos':
                column_mapping[col] = 'bp'
            elif 'ref' in col_lower and 'allele' in col_lower:
                column_mapping[col] = 'ref_allele'
            elif 'alt' in col_lower and 'allele' in col_lower:
                column_mapping[col] = 'alt_allele'
        
        master_snps = master_snps.rename(columns=column_mapping)
    
    # Ensure bp column is integer
    master_snps['bp'] = pd.to_numeric(master_snps['bp'], errors='coerce').astype('Int64')
    
    # Keep only required columns
    master_snps = master_snps[['chr', 'rsid', 'bp', 'ref_allele', 'alt_allele']]
    
    elapsed = time.time() - start_time
    print(f"Master SNP file loaded in {elapsed:.2f} seconds")
    print(f"  Total SNPs: {len(master_snps):,}")
    print(f"  Chromosomes: {sorted(master_snps['chr'].unique())}")
    
    return master_snps


def load_chromosome_variant_file(chr_num, base_path):
    """
    Load chromosome-specific variant file
    
    Returns:
        DataFrame with columns: rsid, chr, bp, ref_allele, alt_allele
    """
    chr_dir = os.path.join(base_path, f"chr{chr_num}")
    variant_file = os.path.join(chr_dir, f"chr{chr_num}_variants.gen")
    
    if not os.path.exists(variant_file):
        raise FileNotFoundError(f"Chromosome variant file not found: {variant_file}")
    
    print(f"\n  Loading chr{chr_num} variant file...")
    
    # Check if file has header
    with open(variant_file, 'r') as f:
        first_line = f.readline().strip()
        has_header = any(keyword in first_line.lower() 
                        for keyword in ['rsid', 'position', 'chr', 'ref', 'alt', 'allele'])
    
    # Load the chromosome variant file
    chr_variants = pd.read_csv(
        variant_file,
        sep=r'\s+',
        header=0 if has_header else None,
        names=None if has_header else ['rsid', 'chr', 'bp', 'ref_allele', 'alt_allele'],
        dtype={'rsid': str, 'chr': str, 'ref_allele': str, 'alt_allele': str}
    )
    
    # Standardize column names
    if has_header:
        column_mapping = {}
        for col in chr_variants.columns:
            col_lower = col.lower()
            if 'rsid' in col_lower or col_lower == 'snp' or col_lower == 'id':
                column_mapping[col] = 'rsid'
            elif col_lower == 'chr' or col_lower == 'chromosome':
                column_mapping[col] = 'chr'
            elif 'position' in col_lower or col_lower == 'bp' or col_lower == 'pos':
                column_mapping[col] = 'bp'
            elif 'ref' in col_lower and 'allele' in col_lower:
                column_mapping[col] = 'ref_allele'
            elif 'alt' in col_lower and 'allele' in col_lower:
                column_mapping[col] = 'alt_allele'
        
        chr_variants = chr_variants.rename(columns=column_mapping)
    
    # Ensure bp column is integer
    chr_variants['bp'] = pd.to_numeric(chr_variants['bp'], errors='coerce').astype('Int64')
    
    # Keep only required columns
    chr_variants = chr_variants[['rsid', 'chr', 'bp', 'ref_allele', 'alt_allele']]
    
    print(f"    Loaded {len(chr_variants):,} SNPs")
    
    return chr_variants


def compute_filtered_indices(chr_num, chr_variants, master_snps):
    """
    Compute which SNP indices to keep for a chromosome
    
    Args:
        chr_num: Chromosome number
        chr_variants: DataFrame of chromosome variants
        master_snps: DataFrame of master SNPs
    
    Returns:
        numpy array of indices to keep
    """
    print(f"  Computing filter indices for chr{chr_num}...")
    start_time = time.time()
    
    # Filter master SNPs for this chromosome
    master_chr = master_snps[master_snps['chr'] == str(chr_num)].copy()
    
    print(f"    Master file has {len(master_chr):,} SNPs for chr{chr_num}")
    
    # Create matching keys (rsid + bp)
    master_chr['match_key'] = master_chr['rsid'].astype(str) + '_' + master_chr['bp'].astype(str)
    chr_variants['match_key'] = chr_variants['rsid'].astype(str) + '_' + chr_variants['bp'].astype(str)
    chr_variants['original_idx'] = range(len(chr_variants))
    
    # Merge to find matches
    merged = chr_variants.merge(
        master_chr[['match_key']],
        on='match_key',
        how='inner'
    )
    
    # Get indices to keep (in original order)
    indices_to_keep = merged['original_idx'].values
    indices_to_keep = np.sort(indices_to_keep)
    
    elapsed = time.time() - start_time
    retention_pct = 100 * len(indices_to_keep) / len(chr_variants) if len(chr_variants) > 0 else 0
    
    print(f"    Computed in {elapsed:.2f} seconds")
    print(f"    Keeping {len(indices_to_keep):,} / {len(chr_variants):,} SNPs ({retention_pct:.2f}%)")
    
    return indices_to_keep


def save_filtered_results(chr_num, indices, chr_variants, base_path, output_formats=['npy', 'gen']):
    """
    Save filtered indices and filtered variant file
    
    Args:
        chr_num: Chromosome number
        indices: Array of indices to keep
        chr_variants: Original chromosome variants DataFrame
        base_path: Base directory path
        output_formats: List of formats to save - 'npy', 'txt', 'gen'
    
    Returns:
        dict with output file paths
    """
    chr_dir = os.path.join(base_path, f"chr{chr_num}")
    output_files = {}
    
    # Save indices (for fast loading in dataloader)
    if 'npy' in output_formats:
        indices_npy = os.path.join(chr_dir, f"chr{chr_num}_filtered_indices_epic.npy")
        np.save(indices_npy, indices)
        output_files['indices_npy'] = indices_npy
        print(f"    Saved indices (binary): {indices_npy}")
    
    if 'txt' in output_formats:
        indices_txt = os.path.join(chr_dir, f"chr{chr_num}_filtered_indices_epic.txt")
        np.savetxt(indices_txt, indices, fmt='%d')
        output_files['indices_txt'] = indices_txt
        print(f"    Saved indices (text): {indices_txt}")
    
    # Save filtered variant file (.gen format)
    if 'gen' in output_formats:
        # Filter the variants DataFrame to keep only matched SNPs
        filtered_variants = chr_variants.iloc[indices].copy()
        
        # Save as .gen file (whitespace-separated, with header)
        filtered_gen = os.path.join(chr_dir, f"chr{chr_num}_variants_filtered_epic.gen")
        
        # Write with tab separation and header
        filtered_variants.to_csv(
            filtered_gen,
            sep='\t',
            index=False,
            header=True
        )
        output_files['filtered_gen'] = filtered_gen
        print(f"    Saved filtered variants: {filtered_gen}")
        print(f"    Filtered .gen contains {len(filtered_variants):,} SNPs")
    
    return output_files


def process_chromosome(chr_num, master_snps, base_path, output_formats=['npy', 'gen']):
    """
    Process a single chromosome: load, filter, save
    
    Args:
        output_formats: List of formats - 'npy', 'txt', 'gen'
    
    Returns:
        dict with statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing Chromosome {chr_num}")
    print(f"{'='*80}")
    
    try:
        # Load chromosome variants
        chr_variants = load_chromosome_variant_file(chr_num, base_path)
        
        # Compute filtered indices
        filtered_indices = compute_filtered_indices(chr_num, chr_variants, master_snps)
        
        # Save filtered results (indices + filtered .gen file)
        output_files = save_filtered_results(
            chr_num, filtered_indices, chr_variants, base_path, output_formats
        )
        
        stats = {
            'chr': chr_num,
            'original_snps': len(chr_variants),
            'filtered_snps': len(filtered_indices),
            'retention_pct': 100 * len(filtered_indices) / len(chr_variants) if len(chr_variants) > 0 else 0,
            'output_files': output_files,
            'success': True
        }
        
        print(f"\n Chromosome {chr_num} processed successfully")
        
    except Exception as e:
        print(f"\n Error processing chromosome {chr_num}: {e}")
        import traceback
        traceback.print_exc()
        
        stats = {
            'chr': chr_num,
            'success': False,
            'error': str(e)
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess SNP filtering for fast training')
    parser.add_argument('-base_path', type=str, default=BASE_PATH, help='Base path for chromosome data')
    parser.add_argument('-master_file', type=str, default=MASTER_SNP_FILE, help='Path to master SNP file')
    parser.add_argument('-output_formats', type=str, default='npy,gen', help='Output formats (comma-separated): npy, txt, gen. Default: npy,gen')
    parser.add_argument('-chromosomes', type=str, default='1-22', help='Chromosomes to process (e.g., "1-22" or "1,2,3")')
    
    args = parser.parse_args()
    
    # Parse output formats
    output_formats = [fmt.strip() for fmt in args.output_formats.split(',')]
    valid_formats = {'npy', 'txt', 'gen'}
    invalid = set(output_formats) - valid_formats
    if invalid:
        print(f"Error: Invalid output formats: {invalid}")
        print(f"Valid formats: {valid_formats}")
        return 1
    
    print("\n" + "="*80)
    print("SNP FILTERING PREPROCESSING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Base path: {args.base_path}")
    print(f"  Master file: {args.master_file}")
    print(f"  Output formats: {', '.join(output_formats)}")
    print(f"  Chromosomes: {args.chromosomes}")
    
    # Parse chromosome list
    if '-' in args.chromosomes:
        start, end = map(int, args.chromosomes.split('-'))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]
    
    print(f"  Processing {len(chromosomes)} chromosomes: {chromosomes}")
    
    # Load master SNP file
    print(f"\n{'='*80}")
    print("Loading Master SNP File")
    print(f"{'='*80}")
    
    master_snps = load_master_snp_file(args.master_file)
    
    # Process each chromosome
    all_stats = []
    total_start = time.time()
    
    for chr_num in chromosomes:
        stats = process_chromosome(chr_num, master_snps, args.base_path, output_formats)
        all_stats.append(stats)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    
    successful = [s for s in all_stats if s.get('success', False)]
    failed = [s for s in all_stats if not s.get('success', False)]
    
    if successful:
        print(f"\n Successfully processed {len(successful)} chromosomes:")
        print(f"\n{'Chr':<5} {'Original SNPs':<15} {'Filtered SNPs':<15} {'Retention %':<15}")
        print("-" * 60)
        
        total_original = 0
        total_filtered = 0
        
        for stats in successful:
            total_original += stats['original_snps']
            total_filtered += stats['filtered_snps']
            print(f"{stats['chr']:<5} {stats['original_snps']:<15,} "
                  f"{stats['filtered_snps']:<15,} {stats['retention_pct']:<15.2f}")
        
        print("-" * 60)
        overall_retention = 100 * total_filtered / total_original if total_original > 0 else 0
        print(f"{'Total':<5} {total_original:<15,} {total_filtered:<15,} {overall_retention:<15.2f}")
    
    if failed:
        print(f"\n Failed to process {len(failed)} chromosomes:")
        for stats in failed:
            print(f"  Chr {stats['chr']}: {stats.get('error', 'Unknown error')}")
    
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds")
    
    if successful:
        print(f"\n{'='*80}")
        print("FILES CREATED")
        print(f"{'='*80}")
        
        print("\nFor each chromosome, the following files were created:")
        if 'npy' in output_formats:
            print("   chr{N}_filtered_indices_epic.npy - Binary indices (fast loading)")
        if 'txt' in output_formats:
            print("   chr{N}_filtered_indices_epic.txt - Text indices (human-readable)")
        if 'gen' in output_formats:
            print("   chr{N}_variants_filtered_epic.gen - Filtered variant file")
        
        print(f"\nExample locations:")
        example_chr = successful[0]['chr']
        for format_type, file_path in successful[0].get('output_files', {}).items():
            print(f"  {file_path}")
        
        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}")
        print("\n1. Filtered files created successfully!")
        print("\n2. Update your dataloader to use prefiltered indices:")
        print("   from ukbb_dataloader_with_prefiltered_indices import prepare_data_splits")
        
        print("\n3. Run your training - data loading will now be much faster!")
        
        print(f"\n Preprocessing complete!")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())