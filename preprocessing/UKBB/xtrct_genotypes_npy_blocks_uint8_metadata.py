#!/usr/bin/env python3
"""
Script to extract variant metadata from BGEN files.
Extracts: rsid, chromosome, position, ref_allele, alt_allele for all variants.
Creates per-chromosome metadata files (.gen format: tab-separated).

TRUE single-pass: reads each variant exactly once to extract metadata.
No genotype data is extracted or stored.
"""

import os
import argparse
from pathlib import Path
from bgen_reader import open_bgen
import pandas as pd
from tqdm import tqdm
import time


def extract_chromosome_metadata(chromosome, base_path, file_pattern, metadata_output_dir):
    """
    Extract variant metadata from a single chromosome BGEN file.
    Single-pass through variants collecting: rsid, position, alleles.
    
    Parameters:
    -----------
    chromosome : int
        Chromosome number
    base_path : str
        Base directory containing BGEN files
    file_pattern : str
        File naming pattern with {} for chromosome
    metadata_output_dir : str
        Output directory for metadata files
    
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
        
        print(f"Extracting metadata for chromosome {chromosome}...")
        
        # Create output directory for this chromosome
        metadata_dir = Path(metadata_output_dir) / f"chr{chromosome}"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Open BGEN file and extract metadata
        with open_bgen(str(filepath)) as bgen:
            n_variants = bgen.nvariants
            
            print(f"  Chromosome {chromosome}: {n_variants:,} variants")
            print(f"  Extracting metadata (single-pass)...")
            
            # Initialize metadata collection
            metadata = {
                'rsid': [],
                'chromosome': [],
                'position': [],
                'ref_allele': [],
                'alt_allele': []
            }
            
            # Single-pass through all variants to extract metadata
            for var_idx in tqdm(range(n_variants), desc=f"Chr{chromosome}", unit="variant"):
                try:
                    # Extract variant metadata
                    rsid = bgen.rsids[var_idx] if hasattr(bgen, 'rsids') else f"var_{var_idx}"
                    position = int(bgen.positions[var_idx]) if hasattr(bgen, 'positions') else 0
                    
                    # Extract alleles
                    if hasattr(bgen, 'allele_ids'):
                        alleles_str = str(bgen.allele_ids[var_idx])
                        # Alleles come as comma-separated string (e.g., 'AC,A')
                        if ',' in alleles_str:
                            alleles = alleles_str.split(',')
                            ref_allele = alleles[0] if len(alleles) > 0 else 'N/A'
                            alt_allele = alleles[1] if len(alleles) > 1 else 'N/A'
                        else:
                            ref_allele = alleles_str
                            alt_allele = 'N/A'
                    else:
                        ref_allele = 'N/A'
                        alt_allele = 'N/A'
                    
                    # Store metadata
                    metadata['rsid'].append(rsid)
                    metadata['chromosome'].append(chromosome)
                    metadata['position'].append(position)
                    metadata['ref_allele'].append(ref_allele)
                    metadata['alt_allele'].append(alt_allele)
                    
                except Exception as e:
                    # Fallback if metadata extraction fails for a variant
                    metadata['rsid'].append(f"var_{var_idx}")
                    metadata['chromosome'].append(chromosome)
                    metadata['position'].append(0)
                    metadata['ref_allele'].append('N/A')
                    metadata['alt_allele'].append('N/A')
            
            # Save metadata to file
            print(f"  Saving metadata file...")
            metadata_df = pd.DataFrame(metadata)
            metadata_file = metadata_dir / f"chr{chromosome}_variants.gen"
            metadata_df.to_csv(metadata_file, sep='\t', index=False)
            
            print(f"  ✓ Chromosome {chromosome} metadata completed!")
            
            return {
                'chromosome': chromosome,
                'status': 'success',
                'n_variants': n_variants,
                'output_dir': str(metadata_dir),
                'metadata_file': str(metadata_file),
                'message': 'Metadata extraction completed successfully'
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
        description='Extract variant metadata from BGEN files (single-pass)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chromosomes
  %(prog)s -path /data/bgen -metadata_output /data/metadata
  
  # Process specific chromosomes
  %(prog)s -path /data/bgen -metadata_output /data/metadata -chr 1 2 3
    
Output Structure (metadata):
  metadata_output_dir/
    chr1/
      chr1_variants.gen      (rsid, chromosome, position, ref_allele, alt_allele)
    chr2/
      chr2_variants.gen
    ...

Notes:
  - Single-pass extraction for memory efficiency
  - Tab-separated .gen format with variant-to-position mapping
        """
    )
        
    parser.add_argument('-path', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra', help='Path to directory containing BGEN files')
    parser.add_argument('-pattern', type=str, default='ukb_imp_chr{}_maf0.05.bgen', help='File naming pattern with {} for chromosome')
    parser.add_argument('-metadata_output', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/variants_metadata_chr', help='Output directory for metadata files (.gen files)')
    parser.add_argument('-chr', nargs='+', type=int, metavar='CHR', help='Specific chromosomes to process (default: all 1-22)')
    
    args = parser.parse_args()
    
    # Determine chromosomes to process
    chromosomes = args.chr if args.chr else list(range(1, 23))
    
    print("=" * 80)
    print("BGEN Metadata Extraction (Single-Pass)")
    print("=" * 80)
    print(f"Input directory:      {args.path}")
    print(f"Metadata output:      {args.metadata_output}")
    print(f"Chromosomes:          {chromosomes}")
    print(f"File pattern:         {args.pattern}")
    print("=" * 80)
    print()
    
    # Create output directories
    metadata_output_dir = Path(args.metadata_output)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chromosomes sequentially
    start_time = time.time()
    results = []
    
    for chr_num in chromosomes:
        print(f"\n{'=' * 80}")
        print(f"Processing Chromosome {chr_num}")
        print('=' * 80)
        result = extract_chromosome_metadata(chr_num, args.path, args.pattern, args.metadata_output)
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
            print(f"✓ Chromosome {result['chromosome']:2d}: {result['n_variants']:8,} variants")
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
        print(f"Output directory: {args.metadata_output}")
    
    print()
    print("=" * 80)
    
    # Save summary to file
    if successful or failed or missing:
        summary_df = pd.DataFrame(results)
        summary_file = metadata_output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Processing summary saved to: {summary_file}")
    
    print()


if __name__ == "__main__":
    main()