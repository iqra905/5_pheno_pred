#!/usr/bin/env python3
"""
Simple script to preview BGEN file contents
Usage: python preview_bgen.py <chromosome_number>
Example: python preview_bgen.py 1
"""

import sys
from bgen_reader import open_bgen
import numpy as np


def preview_bgen_file(filepath, n_variants=5, n_samples=5):
    """Display a preview of BGEN file data"""
    
    print(f"\n{'='*80}")
    print(f"PREVIEW OF: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        with open_bgen(filepath, verbose=False) as bgen:
            # File summary
            print(f"File Summary:")
            print(f"  Total Variants: {bgen.nvariants:,}")
            print(f"  Total Samples:  {bgen.nsamples:,}")
            print()
            
            # Sample IDs (if available)
            if hasattr(bgen, 'samples') and bgen.samples is not None:
                print(f"First {min(n_samples, len(bgen.samples))} Sample IDs:")
                for i, sample_id in enumerate(bgen.samples[:n_samples]):
                    print(f"  Sample {i+1}: {sample_id}")
                print()
            else:
                print("Sample IDs: Not available in file")
                print()
            
            # Get variants dataframe
            variants_df = bgen.variants()
            
            print(f"First {n_variants} Variants:")
            print(f"{'-'*80}")
            
            # Display first N variants
            for idx in range(min(n_variants, len(variants_df))):
                row = variants_df.iloc[idx]
                print(f"\n┌─ Variant {idx+1} " + "─" * 66)
                print(f"│ rsID:       {row['rsid']}")
                print(f"│ Chromosome: {row['chrom']}")
                print(f"│ Position:   {row['pos']:,} bp")
                print(f"│ Ref Allele: {row['a0']}")
                print(f"│ Alt Allele: {row['a1']}")
                print(f"│")
                
                # Read genotype probabilities for this variant
                geno_probs = bgen.read(idx)
                probs = geno_probs.probabilities
                
                print(f"│ Genotype Probabilities (first {min(n_samples, len(probs))} samples):")
                print(f"│   Format: [P({row['a0']}{row['a0']}), P({row['a0']}{row['a1']}), P({row['a1']}{row['a1']})] → Dosage")
                print(f"│")
                
                for i in range(min(n_samples, len(probs))):
                    sample_label = f"Sample {i+1}"
                    if hasattr(bgen, 'samples') and bgen.samples is not None:
                        sample_label = bgen.samples[i]
                    
                    # Get probabilities
                    p_aa, p_ab, p_bb = probs[i]
                    
                    # Calculate dosage (number of alternate alleles)
                    dosage = p_ab + 2 * p_bb
                    
                    # Determine most likely genotype
                    max_prob = max(p_aa, p_ab, p_bb)
                    if max_prob == p_aa:
                        likely_geno = f"{row['a0']}{row['a0']}"
                    elif max_prob == p_ab:
                        likely_geno = f"{row['a0']}{row['a1']}"
                    else:
                        likely_geno = f"{row['a1']}{row['a1']}"
                    
                    print(f"│   {sample_label:15s}: [{p_aa:.3f}, {p_ab:.3f}, {p_bb:.3f}] → {dosage:.3f} (likely: {likely_geno})")
                
                # Summary statistics
                all_dosages = probs[:, 1] + 2 * probs[:, 2]
                print(f"│")
                print(f"│ Dosage Statistics (all {len(all_dosages):,} samples):")
                print(f"│   Mean: {all_dosages.mean():.4f}")
                print(f"│   Std:  {all_dosages.std():.4f}")
                print(f"│   Min:  {all_dosages.min():.4f}")
                print(f"│   Max:  {all_dosages.max():.4f}")
                
                # Allele frequency
                allele_freq = all_dosages.mean() / 2  # Divide by 2 to get frequency
                print(f"│   Alt Allele Frequency: {allele_freq:.4f} ({allele_freq*100:.2f}%)")
                print(f"└" + "─" * 79)
            
            print(f"\n{'='*80}\n")
            
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    return True


def main():
    # Default settings
    base_dir = "."
    file_pattern = "ukb_imp_chr{}_maf0.05.bgen"
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python preview_bgen.py <chromosome_number>")
        print("Example: python preview_bgen.py 1")
        print()
        print("Optional arguments:")
        print("  python preview_bgen.py <chr> <n_variants> <n_samples>")
        print("  python preview_bgen.py 1 10 5  # Show 10 variants, 5 samples each")
        sys.exit(1)
    
    chromosome = sys.argv[1]
    n_variants = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Construct filename
    filename = file_pattern.format(chromosome)
    
    print(f"Looking for file: {filename}")
    print(f"In directory: {base_dir}")
    print()
    
    # Preview the file
    preview_bgen_file(filename, n_variants=n_variants, n_samples=n_samples)


if __name__ == "__main__":
    main()