#!/usr/bin/env python3
"""
Script to extract genotype probabilities from BGEN files and organize by sample.
Creates individual .npy files per sample per chromosome and metadata files.
TRUE single-pass: reads each variant exactly once.
"""

import os
import argparse
from pathlib import Path
from bgen_reader import open_bgen
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def process_chromosome(chromosome, base_path, file_pattern, output_dir, variant_block_size=5000):
    """
    Process a single chromosome with true single-pass through variants.
    Processes variants in blocks and appends to all sample files.
    
    Parameters:
    -----------
    chromosome : int
        Chromosome number
    base_path : str
        Base directory containing BGEN files
    file_pattern : str
        File naming pattern with {} for chromosome
    output_dir : str
        Output directory for extracted data
    variant_block_size : int
        Number of variants to process at once (default: 5000)
    
    Returns:
    --------
    dict : Summary statistics for this chromosome
    """
    try:
        # Build file path
        filepath = Path(base_path) / file_pattern.format(chromosome)
        print(f"Filepath is: {filepath}")

        if not filepath.exists():
            return {
                'chromosome': chromosome,
                'status': 'missing',
                'message': f"File not found: {filepath}"
            }
        
        print(f"Processing chromosome {chromosome}...")
        
        # Create output directory for this chromosome
        chr_output_dir = Path(output_dir) / f"chr{chromosome}"
        chr_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open BGEN file
        with open_bgen(str(filepath)) as bgen:
            n_variants = bgen.nvariants
            n_samples = bgen.nsamples
            
            print(f"  Chromosome {chromosome}: {n_variants:,} variants, {n_samples:,} samples")
            print(f"  Processing in blocks of {variant_block_size:,} variants")
            
            # Estimate memory usage
            memory_per_block_gb = (n_samples * variant_block_size * 3 * 2) / (1024**3)
            print(f"  Estimated memory per block: ~{memory_per_block_gb:.2f} GB")
            
            # Get sample IDs
            try:
                sample_ids = bgen.samples
                if sample_ids is None or len(sample_ids) == 0:
                    raise ValueError("Empty samples")
                print(f"  Loaded {len(sample_ids)} sample IDs from BGEN file")
            except (KeyError, AttributeError, ValueError) as e:
                sample_ids = [f"sample_{i}" for i in range(n_samples)]
                print(f"  Warning: Using generic sample IDs")
            
            # Collect and save metadata
            print(f"  Loading metadata...")
            metadata = {
                'rsid': [],
                'chromosome': [],
                'position': [],
                'ref_allele': [],
                'alt_allele': []
            }
            
            all_rsids = bgen.rsids
            all_positions = bgen.positions
            all_alleles = bgen.allele_ids if hasattr(bgen, 'allele_ids') else None
            
            for var_idx in range(n_variants):
                metadata['rsid'].append(all_rsids[var_idx])
                metadata['chromosome'].append(chromosome)
                metadata['position'].append(int(all_positions[var_idx]))
                
                if all_alleles is not None:
                    alleles = all_alleles[var_idx]
                    metadata['ref_allele'].append(alleles[0] if len(alleles) > 0 else 'N/A')
                    metadata['alt_allele'].append(alleles[1] if len(alleles) > 1 else 'N/A')
                else:
                    metadata['ref_allele'].append('N/A')
                    metadata['alt_allele'].append('N/A')
            
            print(f"  Saving metadata file...")
            metadata_df = pd.DataFrame(metadata)
            metadata_file = chr_output_dir / f"chr{chromosome}_variants.gen"
            metadata_df.to_csv(metadata_file, sep='\t', index=False)
            
            # Initialize all sample files with headers
            print(f"  Initializing {n_samples:,} sample files...")
            sample_files = []
            for sample_idx in tqdm(range(n_samples), desc="Creating files", leave=False):
                sample_id = sample_ids[sample_idx]
                sample_file = chr_output_dir / f"sample_{sample_id}_chr{chromosome}.npy"
                sample_files.append(sample_file)
                
                # Create NPY file with header
                with open(sample_file, 'wb') as f:
                    header = {
                        'descr': np.dtype(np.float16).descr[0][1],
                        'fortran_order': False,
                        'shape': (n_variants, 3)
                    }
                    np.lib.format.write_array_header_1_0(f, header)
            
            # Process variants in blocks - TRUE SINGLE PASS
            n_blocks = (n_variants + variant_block_size - 1) // variant_block_size
            print(f"  Processing {n_variants:,} variants in {n_blocks} blocks (SINGLE PASS)...")
            
            for block_idx in tqdm(range(n_blocks), desc=f"Chr{chromosome}"):
                start_var = block_idx * variant_block_size
                end_var = min(start_var + variant_block_size, n_variants)
                block_n_variants = end_var - start_var
                
                # Read this block of variants into memory
                # Shape: (n_samples, block_n_variants, 3)
                block_data = np.zeros((n_samples, block_n_variants, 3), dtype=np.float16)
                
                for i, var_idx in enumerate(range(start_var, end_var)):
                    probs = bgen.read(var_idx)
                    
                    # Handle different shapes
                    if len(probs.shape) == 3 and probs.shape[1] == 1:
                        probs = probs.squeeze(axis=1)
                    
                    block_data[:, i, :] = probs.astype(np.float16)
                
                # Append this block to ALL sample files
                for sample_idx in range(n_samples):
                    sample_data = block_data[sample_idx, :, :]  # Shape: (block_n_variants, 3)
                    
                    with open(sample_files[sample_idx], 'ab') as f:
                        f.write(sample_data.tobytes())
                
                # Free memory
                del block_data
            
            print(f"  ✓ Chromosome {chromosome} completed!")
            
            return {
                'chromosome': chromosome,
                'status': 'success',
                'n_variants': n_variants,
                'n_samples': n_samples,
                'output_dir': str(chr_output_dir),
                'message': 'Completed successfully'
            }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"  ✗ Error processing chromosome {chromosome}:")
        print(error_details)
        return {
            'chromosome': chromosome,
            'status': 'error',
            'message': f"Error: {str(e)}\n{error_details}"
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract genotype probabilities by sample from BGEN files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chromosomes
  %(prog)s -path /data/bgen -output /data/output
  
  # Process specific chromosomes
  %(prog)s -path /data/bgen -output /data/output -chr 1 2 3
  
  # Adjust block size (smaller = less memory, slower)
  %(prog)s -path /data/bgen -output /data/output -chr 1 -blocksize 2000
    
Output Structure:
  output_dir/
    chr1/
      sample_<id>_chr1.npy  (n_variants x 3 array of float16)
      chr1_variants.gen      (metadata file)
    chr2/
      sample_<id>_chr2.npy
      chr2_variants.gen
    ...
        """
    )
        
    parser.add_argument('-path', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra', help='Path to directory containing BGEN files')
    parser.add_argument('-pattern', type=str, default='ukb_imp_chr{}_maf0.05.bgen', help='File naming pattern with {} for chromosome')
    parser.add_argument('-output', type=str, default='/mnt/fast/nobackup/scratch4weeks/if00208/ukbb/genotypes_chr', help='Output directory for extracted data')
    parser.add_argument('-chr', nargs='+', type=int, metavar='CHR', help='Specific chromosomes to process (default: all 1-22)')
    parser.add_argument('-blocksize', type=int, default=5000, help='Number of variants per block (default: 5000)')
    
    args = parser.parse_args()
    
    # Determine chromosomes to process
    chromosomes = args.chr if args.chr else list(range(1, 23))
    
    print("=" * 80)
    print("BGEN Genotype Extraction by Sample (Block Processing)")
    print("=" * 80)
    print(f"Input directory:  {args.path}")
    print(f"Output directory: {args.output}")
    print(f"Chromosomes:      {chromosomes}")
    print(f"File pattern:     {args.pattern}")
    print(f"Variant block:    {args.blocksize:,} variants")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chromosomes sequentially
    start_time = time.time()
    results = []
    
    for chr_num in chromosomes:
        print(f"\n{'=' * 80}")
        print(f"Processing Chromosome {chr_num}")
        print('=' * 80)
        result = process_chromosome(chr_num, args.path, args.pattern, args.output, args.blocksize)
        results.append(result)
        print()
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print()
    print("=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    successful = []
    failed = []
    missing = []
    
    for result in results:
        if result['status'] == 'success':
            successful.append(result)
            print(f"✓ Chromosome {result['chromosome']:2d}: {result['n_variants']:8,} variants, "
                  f"{result['n_samples']:6,} samples")
        elif result['status'] == 'missing':
            missing.append(result)
            print(f"⊘ Chromosome {result['chromosome']:2d}: File not found")
        else:
            failed.append(result)
            print(f"✗ Chromosome {result['chromosome']:2d}: Error occurred")
            print(f"  Details: {result['message'][:200]}")
    
    print()
    print(f"Completed: {len(successful)}/{len(chromosomes)} chromosomes")
    print(f"Missing:   {len(missing)} chromosomes")
    print(f"Failed:    {len(failed)} chromosomes")
    print(f"Time:      {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    if successful:
        total_variants = sum(r['n_variants'] for r in successful)
        print(f"\nTotal variants processed: {total_variants:,}")
        print(f"Output directory: {args.output}")
    
    print()
    print("=" * 80)
    
    # Save summary to file
    if successful or failed or missing:
        summary_df = pd.DataFrame(results)
        summary_file = output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Processing summary saved to: {summary_file}")
    
    print()


if __name__ == "__main__":
    main()