#!/usr/bin/env python3
"""
SNP Filtering Preprocessing Script - Single Nucleotide Variants (SNVs)

This script filters chromosome-wise genotype variant files to keep only
single nucleotide variants (SNVs) where both ref and alt alleles have length 1.

Removes:
  - Indels (length > 1)
  - Multiallelic sites with commas
  - Missing/invalid alleles

Output:
    Creates chr{N}_variants_snv_only.gen files with SNV-only variants
    Creates chr{N}_snv_filtered_indices.npy with SNV indices
    Creates filtering_summary.csv with per-chromosome statistics
"""

import os
import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path
import json


BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/variants_metadata_chr"
OUTPUT_BASE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/snv_filtered"


def load_chromosome_variant_file(chr_num, base_path):
    """
    Load chromosome-specific variant file
    
    Args:
        chr_num: Chromosome number
        base_path: Base directory path
    
    Returns:
        DataFrame with columns: rsid, chr, bp, ref_allele, alt_allele
    """
    chr_dir = os.path.join(base_path, f"chr{chr_num}")
    variant_file = os.path.join(chr_dir, f"chr{chr_num}_variants.gen")
    
    if not os.path.exists(variant_file):
        raise FileNotFoundError(f"Chromosome variant file not found: {variant_file}")
    
    print(f"  Loading chr{chr_num} variant file...")
    
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
    
    print(f"    Loaded {len(chr_variants):,} total SNPs")
    
    return chr_variants


def compute_snv_indices(chr_num, chr_variants):
    """
    Compute which SNP indices are single nucleotide variants (SNVs)
    
    Criteria for SNV:
      - ref_allele length == 1 (single nucleotide)
      - alt_allele length == 1 (single nucleotide)
      - No missing/invalid characters
      - No commas (multiallelic placeholders)
    
    Args:
        chr_num: Chromosome number
        chr_variants: DataFrame of chromosome variants
    
    Returns:
        tuple: (indices_to_keep, stats_dict)
    """
    print(f"  Computing SNV filter indices for chr{chr_num}...")
    start_time = time.time()
    
    # Create mask for valid SNVs
    mask = (
        (chr_variants['ref_allele'].str.len() == 1) &
        (chr_variants['alt_allele'].str.len() == 1) &
        ~(chr_variants['ref_allele'].str.contains(',', na=False)) &
        ~(chr_variants['alt_allele'].str.contains(',', na=False)) &
        (chr_variants['ref_allele'].notna()) &
        (chr_variants['alt_allele'].notna()) &
        (chr_variants['ref_allele'] != '') &
        (chr_variants['alt_allele'] != '')
    )
    
    # Get indices to keep
    indices_to_keep = np.where(mask)[0]
    indices_to_keep = np.sort(indices_to_keep)
    
    elapsed = time.time() - start_time
    
    total_snps = len(chr_variants)
    snv_count = len(indices_to_keep)
    retention_pct = 100 * snv_count / total_snps if total_snps > 0 else 0
    
    print(f"    Computed in {elapsed:.2f} seconds")
    print(f"    Keeping {snv_count:,} / {total_snps:,} SNVs ({retention_pct:.2f}%)")
    
    # Collect filtering statistics
    variant_types = {
        'total': total_snps,
        'snv_kept': snv_count,
        'removed': total_snps - snv_count,
        'retention_pct': retention_pct
    }
    
    # Count removed by category
    multi_allelic = (chr_variants['ref_allele'].str.contains(',', na=False) | 
                     chr_variants['alt_allele'].str.contains(',', na=False)).sum()
    indel = ((chr_variants['ref_allele'].str.len() > 1) | 
             (chr_variants['alt_allele'].str.len() > 1)).sum()
    invalid = (chr_variants['ref_allele'].isna() | chr_variants['alt_allele'].isna() |
               (chr_variants['ref_allele'] == '') | (chr_variants['alt_allele'] == '')).sum()
    
    # Adjust indel count to not double-count
    indel = indel - ((chr_variants['ref_allele'].str.len() > 1) & 
                      chr_variants['ref_allele'].str.contains(',', na=False)).sum()
    
    variant_types['indels_removed'] = indel
    variant_types['multiallelic_removed'] = multi_allelic
    variant_types['invalid_removed'] = invalid
    
    return indices_to_keep, variant_types


def save_filtered_results(chr_num, indices, chr_variants, output_base, output_formats=['npy', 'gen']):
    """
    Save filtered indices and filtered variant file
    
    Args:
        chr_num: Chromosome number
        indices: Array of indices to keep
        chr_variants: Original chromosome variants DataFrame
        output_base: Base output directory
        output_formats: List of formats to save - 'npy', 'txt', 'gen'
    
    Returns:
        dict with output file paths
    """
    chr_dir = os.path.join(output_base, f"chr{chr_num}")
    os.makedirs(chr_dir, exist_ok=True)
    
    output_files = {}
    
    # Save indices (for fast loading in dataloader)
    if 'npy' in output_formats:
        indices_npy = os.path.join(chr_dir, f"chr{chr_num}_snv_filtered_indices.npy")
        np.save(indices_npy, indices)
        output_files['indices_npy'] = indices_npy
        print(f"    Saved indices (binary): {indices_npy}")
    
    if 'txt' in output_formats:
        indices_txt = os.path.join(chr_dir, f"chr{chr_num}_snv_filtered_indices.txt")
        np.savetxt(indices_txt, indices, fmt='%d')
        output_files['indices_txt'] = indices_txt
        print(f"    Saved indices (text): {indices_txt}")
    
    # Save filtered variant file (.gen format)
    if 'gen' in output_formats:
        # Filter the variants DataFrame to keep only SNVs
        filtered_variants = chr_variants.iloc[indices].copy()
        
        # Save as .gen file (tab-separated, with header)
        filtered_gen = os.path.join(chr_dir, f"chr{chr_num}_variants_snv_only.gen")
        
        # Write with tab separation and header
        filtered_variants.to_csv(
            filtered_gen,
            sep='\t',
            index=False,
            header=True
        )
        output_files['filtered_gen'] = filtered_gen
        print(f"    Saved filtered variants: {filtered_gen}")
        print(f"    SNV-only .gen contains {len(filtered_variants):,} SNVs")
    
    return output_files


def process_chromosome(chr_num, base_path, output_base, output_formats=['npy', 'gen']):
    """
    Process a single chromosome: load, filter SNVs, save
    
    Args:
        chr_num: Chromosome number
        base_path: Base path for chromosome data
        output_base: Base output directory
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
        
        # Compute SNV filtered indices
        filtered_indices, variant_types = compute_snv_indices(chr_num, chr_variants)
        
        # Save filtered results (indices + filtered .gen file)
        output_files = save_filtered_results(
            chr_num, filtered_indices, chr_variants, output_base, output_formats
        )
        
        stats = {
            'chr': chr_num,
            'total_snps': variant_types['total'],
            'snv_kept': variant_types['snv_kept'],
            'removed': variant_types['removed'],
            'retention_pct': variant_types['retention_pct'],
            'indels_removed': variant_types.get('indels_removed', 0),
            'multiallelic_removed': variant_types.get('multiallelic_removed', 0),
            'invalid_removed': variant_types.get('invalid_removed', 0),
            'output_files': output_files,
            'success': True
        }
        
        print(f"\n✓ Chromosome {chr_num} processed successfully")
        
    except Exception as e:
        print(f"\n✗ Error processing chromosome {chr_num}: {e}")
        import traceback
        traceback.print_exc()
        
        stats = {
            'chr': chr_num,
            'success': False,
            'error': str(e)
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Filter UKBB SNVs to keep only single nucleotide variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chromosomes, save both binary and .gen format
  %(prog)s
  
  # Process specific chromosomes, save only binary indices
  %(prog)s -chromosomes 1,2,3 -output_formats npy
  
  # Process chromosomes 1-10 with all output formats
  %(prog)s -chromosomes 1-10 -output_formats npy,txt,gen

Variant Filtering Criteria:
  Kept: SNVs with ref_allele length == 1 AND alt_allele length == 1
  Removed: Indels (length > 1), multiallelic (commas), invalid/missing
    
Output Structure (per chromosome):
  snv_filtered/
    chr1/
      chr1_snv_filtered_indices.npy        ← Fast binary index file
      chr1_snv_filtered_indices.txt        ← Human-readable indices
      chr1_variants_snv_only.gen           ← SNV-only variant metadata
    chr2/
      chr2_snv_filtered_indices.npy
      ...
    filtering_summary.csv
        """
    )
    
    parser.add_argument('-base_path', type=str, default=BASE_PATH, 
                        help='Base path for chromosome data')
    parser.add_argument('-output_base', type=str, default=OUTPUT_BASE, 
                        help='Base output directory for filtered SNVs')
    parser.add_argument('-output_formats', type=str, default='npy,gen', 
                        help='Output formats (comma-separated): npy, txt, gen. Default: npy,gen')
    parser.add_argument('-chromosomes', type=str, default='1-22', 
                        help='Chromosomes to process (e.g., "1-22" or "1,2,3")')
    
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
    print("SNP FILTERING PREPROCESSING - SINGLE NUCLEOTIDE VARIANTS (SNVs)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Base path: {args.base_path}")
    print(f"  Output base: {args.output_base}")
    print(f"  Output formats: {', '.join(output_formats)}")
    print(f"  Chromosomes: {args.chromosomes}")
    
    # Parse chromosome list
    if '-' in args.chromosomes:
        start, end = map(int, args.chromosomes.split('-'))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]
    
    print(f"  Processing {len(chromosomes)} chromosomes: {chromosomes}")
    
    # Create output base directory
    output_base_dir = Path(args.output_base)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each chromosome
    all_stats = []
    total_start = time.time()
    
    for chr_num in chromosomes:
        stats = process_chromosome(chr_num, args.base_path, args.output_base, output_formats)
        all_stats.append(stats)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n\n" + "="*80)
    print("SNV FILTERING SUMMARY")
    print("="*80)
    
    successful = [s for s in all_stats if s.get('success', False)]
    failed = [s for s in all_stats if not s.get('success', False)]
    
    if successful:
        print(f"\n✓ Successfully processed {len(successful)} chromosomes:")
        print(f"\n{'Chr':<5} {'Total SNPs':<15} {'SNVs Kept':<15} {'Retention %':<12} {'Indels':<12} {'Multi':<12}")
        print("-" * 80)
        
        total_original = 0
        total_snv = 0
        total_indels = 0
        total_multiallelic = 0
        
        for stats in successful:
            total_original += stats['total_snps']
            total_snv += stats['snv_kept']
            total_indels += stats.get('indels_removed', 0)
            total_multiallelic += stats.get('multiallelic_removed', 0)
            print(f"{stats['chr']:<5} {stats['total_snps']:<15,} "
                  f"{stats['snv_kept']:<15,} {stats['retention_pct']:<12.2f} "
                  f"{stats.get('indels_removed', 0):<12,} "
                  f"{stats.get('multiallelic_removed', 0):<12,}")
        
        print("-" * 80)
        overall_retention = 100 * total_snv / total_original if total_original > 0 else 0
        print(f"{'Total':<5} {total_original:<15,} {total_snv:<15,} {overall_retention:<12.2f} "
              f"{total_indels:<12,} {total_multiallelic:<12,}")
    
    if failed:
        print(f"\n✗ Failed to process {len(failed)} chromosomes:")
        for stats in failed:
            print(f"  Chr {stats['chr']}: {stats.get('error', 'Unknown error')}")
    
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    # Save summary to CSV
    if successful:
        print(f"\n{'='*80}")
        print("CREATING SUMMARY FILES")
        print(f"{'='*80}")
        
        summary_df = pd.DataFrame(successful)
        # Select columns to keep in summary
        summary_cols = ['chr', 'total_snps', 'snv_kept', 'removed', 'retention_pct', 
                        'indels_removed', 'multiallelic_removed', 'invalid_removed']
        summary_df = summary_df[[col for col in summary_cols if col in summary_df.columns]]
        
        summary_file = output_base_dir / "filtering_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        # Save detailed JSON summary
        summary_json_data = {
            'description': 'SNV filtering results - keeping only single nucleotide variants',
            'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'filter_criteria': {
                'ref_allele_length': 1,
                'alt_allele_length': 1,
                'no_commas': True,
                'no_missing': True
            },
            'chromosomes_processed': len(successful),
            'totals': {
                'total_variants': int(total_original),
                'snv_kept': int(total_snv),
                'retention_pct': round(overall_retention, 2),
                'indels_removed': int(total_indels),
                'multiallelic_removed': int(total_multiallelic)
            },
            'per_chromosome': [
                {
                    'chr': s['chr'],
                    'total_snps': s['total_snps'],
                    'snv_kept': s['snv_kept'],
                    'retention_pct': s['retention_pct'],
                    'indels_removed': s.get('indels_removed', 0),
                    'multiallelic_removed': s.get('multiallelic_removed', 0)
                }
                for s in successful
            ]
        }
        
        summary_json = output_base_dir / "filtering_summary.json"
        with open(summary_json, 'w') as f:
            json.dump(summary_json_data, f, indent=2)
        print(f"Detailed summary saved to: {summary_json}")
        
        # Print next steps
        print(f"\n{'='*80}")
        print("FILES CREATED")
        print(f"{'='*80}")
        
        print("\nFor each chromosome, the following files were created:")
        if 'npy' in output_formats:
            print("   chr{N}_snv_filtered_indices.npy - Binary indices (fast loading)")
        if 'txt' in output_formats:
            print("   chr{N}_snv_filtered_indices.txt - Text indices (human-readable)")
        if 'gen' in output_formats:
            print("   chr{N}_variants_snv_only.gen - SNV-only variant metadata")
        
        print(f"\nOutput directory: {args.output_base}")
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"✓ Processed {len(successful)} chromosomes successfully")
        print(f"✓ Original variants: {total_original:,}")
        print(f"✓ SNVs kept: {total_snv:,} ({overall_retention:.2f}%)")
        print(f"✓ Variants removed:")
        print(f"    - Indels: {total_indels:,}")
        print(f"    - Multiallelic: {total_multiallelic:,}")
        print(f"✓ Filtering complete!")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
