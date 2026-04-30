#!/usr/bin/env python3
"""
Script to merge per-chromosome sample files into single files per sample.
Combines chr1-chr22 .npy files for each sample and creates a master metadata file.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def get_sample_ids(input_dir, chromosome=1):
    """
    Extract all sample IDs from chromosome files.
    
    Parameters:
    -----------
    input_dir : Path
        Input directory containing chromosome folders
    chromosome : int
        Chromosome to scan for sample IDs (default: 1)
    
    Returns:
    --------
    list : Sample IDs found
    """
    chr_dir = input_dir / f"chr{chromosome}"
    if not chr_dir.exists():
        raise ValueError(f"Chromosome {chromosome} directory not found: {chr_dir}")
    
    sample_ids = []
    pattern = f"sample_*_chr{chromosome}.npy"
    
    for file in chr_dir.glob(pattern):
        # Extract sample ID from filename: sample_XXXX_chr1.npy -> XXXX
        filename = file.stem  # Remove .npy
        parts = filename.split('_')
        if len(parts) >= 3:
            # Handle sample IDs that might contain underscores
            sample_id = '_'.join(parts[1:-1])  # Everything between 'sample_' and '_chrX'
            sample_ids.append(sample_id)
    
    return sorted(sample_ids)


def merge_sample_files(sample_id, input_dir, output_dir, chromosomes):
    """
    Merge all chromosome files for a single sample.
    
    Parameters:
    -----------
    sample_id : str
        Sample ID to process
    input_dir : Path
        Input directory containing chromosome folders
    output_dir : Path
        Output directory for merged files
    chromosomes : list
        List of chromosomes to merge
    
    Returns:
    --------
    dict : Summary statistics
    """
    try:
        merged_data = []
        chromosomes_found = []
        
        for chr_num in chromosomes:
            chr_dir = input_dir / f"chr{chr_num}"
            sample_file = chr_dir / f"sample_{sample_id}_chr{chr_num}.npy"
            
            if sample_file.exists():
                # Load chromosome data
                chr_data = np.load(sample_file)
                merged_data.append(chr_data)
                chromosomes_found.append(chr_num)
            else:
                # Chromosome file missing for this sample
                pass
        
        if not merged_data:
            return {
                'sample_id': sample_id,
                'status': 'missing',
                'chromosomes_found': 0,
                'total_variants': 0,
                'message': 'No chromosome files found'
            }
        
        # Concatenate all chromosomes
        merged_array = np.concatenate(merged_data, axis=0)
        
        # Save merged file
        output_file = output_dir / f"sample_{sample_id}_merged.npy"
        np.save(output_file, merged_array)
        
        return {
            'sample_id': sample_id,
            'status': 'success',
            'chromosomes_found': len(chromosomes_found),
            'chromosomes': chromosomes_found,
            'total_variants': merged_array.shape[0],
            'message': 'Merged successfully'
        }
    
    except Exception as e:
        return {
            'sample_id': sample_id,
            'status': 'error',
            'chromosomes_found': 0,
            'total_variants': 0,
            'message': f"Error: {str(e)}"
        }


def merge_metadata_files(input_dir, output_dir, chromosomes):
    """
    Merge all chromosome metadata files into a single master file.
    
    Parameters:
    -----------
    input_dir : Path
        Input directory containing chromosome folders
    output_dir : Path
        Output directory for merged metadata
    chromosomes : list
        List of chromosomes to merge
    
    Returns:
    --------
    int : Total number of variants in merged metadata
    """
    print("  Merging metadata files...")
    
    metadata_dfs = []
    
    for chr_num in tqdm(chromosomes, desc="Loading metadata", leave=False):
        chr_dir = input_dir / f"chr{chr_num}"
        metadata_file = chr_dir / f"chr{chr_num}_variants.gen"
        
        if metadata_file.exists():
            df = pd.read_csv(metadata_file, sep='\t')
            metadata_dfs.append(df)
        else:
            print(f"    Warning: Metadata file not found for chromosome {chr_num}")
    
    if not metadata_dfs:
        print("    Error: No metadata files found!")
        return 0
    
    # Concatenate all metadata
    merged_metadata = pd.concat(metadata_dfs, ignore_index=True)
    
    # Save merged metadata
    output_file = output_dir / "merged_variants.gen"
    merged_metadata.to_csv(output_file, sep='\t', index=False)
    
    print(f"    Merged metadata saved: {merged_metadata.shape[0]:,} total variants")
    
    return merged_metadata.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-chromosome sample files into single files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all samples for chromosomes 1-22
  %(prog)s -input /data/genotype_data_npy -output /data/merged_samples
  
  # Merge specific chromosomes only
  %(prog)s -input /data/genotype_data_npy -output /data/merged_samples -chr 1 2 3
  
  # Process specific samples only
  %(prog)s -input /data/genotype_data_npy -output /data/merged_samples -samples sample_123 sample_456

Input Structure (expected):
  input_dir/
    chr1/
      sample_<id>_chr1.npy
      chr1_variants.gen
    chr2/
      sample_<id>_chr2.npy
      chr2_variants.gen
    ...

Output Structure:
  output_dir/
    sample_<id>_merged.npy    (all variants x 3 array)
    merged_variants.gen        (master metadata file)
    merge_summary.csv          (processing summary)
        """
    )
    
    parser.add_argument('-input', type=str, required=True, 
                        help='Input directory containing chromosome folders')
    parser.add_argument('-output', type=str, required=True,
                        help='Output directory for merged files')
    parser.add_argument('-chr', nargs='+', type=int, metavar='CHR',
                        help='Specific chromosomes to merge (default: 1-22)')
    parser.add_argument('-samples', nargs='+', type=str, metavar='SAMPLE_ID',
                        help='Specific samples to process (default: all samples)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine chromosomes to merge
    chromosomes = args.chr if args.chr else list(range(1, 23))
    chromosomes = sorted(chromosomes)  # Ensure they're in order
    
    print("=" * 80)
    print("Sample File Merger - Combining Chromosomes")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Chromosomes:      {chromosomes}")
    print("=" * 80)
    print()
    
    # Merge metadata files first
    print("Step 1: Merging metadata files")
    print("-" * 80)
    total_variants = merge_metadata_files(input_dir, output_dir, chromosomes)
    print()
    
    # Get sample IDs
    print("Step 2: Identifying samples")
    print("-" * 80)
    
    if args.samples:
        sample_ids = args.samples
        print(f"  Processing {len(sample_ids):,} specified samples")
    else:
        print(f"  Scanning chromosome {chromosomes[0]} for sample IDs...")
        sample_ids = get_sample_ids(input_dir, chromosome=chromosomes[0])
        print(f"  Found {len(sample_ids):,} samples")
    
    print()
    
    # Process samples
    print("Step 3: Merging sample files")
    print("-" * 80)
    print(f"  Processing {len(sample_ids):,} samples across {len(chromosomes)} chromosomes...")
    print()
    
    start_time = time.time()
    results = []
    
    for sample_id in tqdm(sample_ids, desc="Merging samples"):
        result = merge_sample_files(sample_id, input_dir, output_dir, chromosomes)
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print()
    print("=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    missing = [r for r in results if r['status'] == 'missing']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"Successful: {len(successful):,} samples")
    print(f"Missing:    {len(missing):,} samples (no chromosome files found)")
    print(f"Failed:     {len(failed):,} samples")
    print(f"Time:       {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    if successful:
        total_variants_check = successful[0]['total_variants']
        print(f"\nVariants per sample: {total_variants_check:,}")
        print(f"Expected variants:   {total_variants:,}")
        
        if total_variants_check != total_variants:
            print(f"  Warning: Variant count mismatch detected!")
        
        # Check for incomplete merges
        incomplete = [r for r in successful if r['chromosomes_found'] < len(chromosomes)]
        if incomplete:
            print(f"\nWarning: {len(incomplete):,} samples have incomplete chromosome data:")
            for r in incomplete[:5]:  # Show first 5
                missing_chrs = set(chromosomes) - set(r['chromosomes'])
                print(f"  Sample {r['sample_id']}: missing chromosomes {sorted(missing_chrs)}")
            if len(incomplete) > 5:
                print(f"  ... and {len(incomplete) - 5} more")
    
    print(f"\nOutput directory: {output_dir}")
    print()
    print("=" * 80)
    
    # Save detailed summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / "merge_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Detailed summary saved to: {summary_file}")
    
    # Save list of successfully merged samples
    if successful:
        successful_samples = [r['sample_id'] for r in successful]
        samples_file = output_dir / "merged_samples_list.txt"
        with open(samples_file, 'w') as f:
            f.write('\n'.join(successful_samples))
        print(f"List of merged samples saved to: {samples_file}")
    
    print()


if __name__ == "__main__":
    main()