#!/usr/bin/env python
"""
Merge GWAS results from multiple jobs/SNP ranges.

This script combines results from multiple jobs that each processed
different SNP ranges of the same chromosome, or different chromosomes.
It also provides summary statistics and identifies genome-wide significant SNPs.
"""

import pandas as pd
import glob
import os
import sys
from pathlib import Path
import argparse
import numpy as np

def merge_gwas_results(output_dir, p_threshold=5e-8, verbose=True):
    """
    Merge all GWAS result files in output directory.
    
    Args:
        output_dir: Directory containing CSV files with GWAS results
        p_threshold: Genome-wide significance threshold (default: 5e-8)
        verbose: Print summary statistics
    
    Returns:
        Tuple of (all_snps_df, significant_snps_df, summary_stats)
    """
    
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Find all significant SNP files
    sig_pattern = os.path.join(output_dir, "train_set_significant_snps_chr*.csv")
    sig_files = sorted(glob.glob(sig_pattern))
    
    # Find all SNP files
    all_pattern = os.path.join(output_dir, "train_set_all_snps_chr*.csv")
    all_files = sorted(glob.glob(all_pattern))
    
    # Find failed SNP files
    failed_pattern = os.path.join(output_dir, "train_set_failed_snps_chr*.csv")
    failed_files = sorted(glob.glob(failed_pattern))
    
    if verbose:
        print(f"Found {len(sig_files)} significant SNP files")
        print(f"Found {len(all_files)} all SNP files")
        print(f"Found {len(failed_files)} failed SNP files")
    
    # Merge significant results
    if sig_files:
        sig_dfs = []
        for f in sig_files:
            try:
                df = pd.read_csv(f)
                sig_dfs.append(df)
                if verbose:
                    print(f"  Loaded {f}: {len(df)} SNPs")
            except Exception as e:
                print(f"  WARNING: Could not load {f}: {e}")
        
        if sig_dfs:
            sig_merged = pd.concat(sig_dfs, ignore_index=True)
            sig_merged = sig_merged.drop_duplicates(subset=['SNP_Index', 'Chromosome'])
            sig_merged = sig_merged.sort_values(['Chromosome', 'SNP_Index'])
            sig_merged.to_csv(
                os.path.join(output_dir, 'ALL_significant_snps_merged.csv'),
                index=False
            )
            if verbose:
                print(f"\n✓ Saved merged significant SNPs: {len(sig_merged)} total")
        else:
            sig_merged = None
    else:
        sig_merged = None
        print(f"No significant SNP files found")
    
    # Merge all results (optional, can be memory intensive)
    if all_files and len(all_files) * 100_000 < 1e9:  # Rough memory check
        if verbose:
            print(f"\nMerging all SNP results (may take a moment)...")
        
        all_dfs = []
        for f in all_files:
            try:
                df = pd.read_csv(f)
                all_dfs.append(df)
            except Exception as e:
                print(f"  WARNING: Could not load {f}: {e}")
        
        if all_dfs:
            all_merged = pd.concat(all_dfs, ignore_index=True)
            all_merged = all_merged.drop_duplicates(subset=['SNP_Index', 'Chromosome'])
            all_merged = all_merged.sort_values(['Chromosome', 'SNP_Index'])
            all_merged.to_csv(
                os.path.join(output_dir, 'ALL_snps_merged.csv'),
                index=False
            )
            if verbose:
                print(f"✓ Saved all SNPs: {len(all_merged)} total")
        else:
            all_merged = None
    else:
        all_merged = None
        if verbose and all_files:
            print(f"Skipping all SNPs merge (too many files/memory)")
    
    # Merge failed SNPs
    if failed_files:
        failed_dfs = []
        for f in failed_files:
            try:
                df = pd.read_csv(f)
                failed_dfs.append(df)
            except Exception as e:
                print(f"  WARNING: Could not load {f}: {e}")
        
        if failed_dfs:
            failed_merged = pd.concat(failed_dfs, ignore_index=True)
            failed_merged = failed_merged.drop_duplicates(subset=['SNP_Index', 'Chromosome'])
            failed_merged.to_csv(
                os.path.join(output_dir, 'ALL_failed_snps.csv'),
                index=False
            )
            if verbose:
                print(f"✓ Saved failed SNPs: {len(failed_merged)} total")
        else:
            failed_merged = None
    else:
        failed_merged = None
    
    # Compute summary statistics
    summary = {}
    if all_merged is not None:
        summary['total_snps'] = len(all_merged)
        summary['total_chromosomes'] = all_merged['Chromosome'].nunique()
        summary['failed_snps'] = all_merged['Failed'].sum()
        summary['tested_snps'] = len(all_merged) - summary['failed_snps']
        summary['significant_snps_p0.05'] = (all_merged['P_Value'] <= 0.05).sum()
        summary['significant_snps_p1e-5'] = (all_merged['P_Value'] <= 1e-5).sum()
        summary['significant_snps_gwas'] = (all_merged['P_Value'] <= p_threshold).sum()
        summary['min_p_value'] = all_merged['P_Value'].min()
        summary['max_beta'] = all_merged['Beta'].abs().max()
    
    if sig_merged is not None:
        summary['gwas_sig_snps'] = len(sig_merged)
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, float):
                if value < 1e-10:
                    print(f"{key:25s}: {value:.3e}")
                elif value < 0.001:
                    print(f"{key:25s}: {value:.3e}")
                else:
                    print(f"{key:25s}: {value:.6f}")
            else:
                print(f"{key:25s}: {value:,}")
    
    return all_merged, sig_merged, summary


def create_manhattan_plot_data(all_snps_df, output_file=None):
    """
    Prepare data for Manhattan plot visualization.
    
    Returns DataFrame with cumulative positions for plotting.
    """
    if all_snps_df is None or all_snps_df.empty:
        return None
    
    # Map chromosome numbers to positions
    chrom_sizes = {
        '1': 249250621, '2': 242193529, '3': 198399188, '4': 190454276,
        '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
        '9': 138394717, '10': 135534747, '11': 135006516, '12': 133851895,
        '13': 115169878, '14': 107349540, '15': 102531392, '16': 90354753,
        '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520,
        '21': 48129895, '22': 51304566
    }
    
    df = all_snps_df.copy()
    df['Chromosome'] = df['Chromosome'].astype(str)
    
    # Calculate cumulative positions
    cumsum = 0
    df['cumsum_start'] = 0
    for chrom in sorted(df['Chromosome'].unique(), key=lambda x: int(x) if x.isdigit() else 999):
        size = chrom_sizes.get(str(chrom), 100_000_000)
        mask = df['Chromosome'] == chrom
        df.loc[mask, 'cumsum_start'] = cumsum
        cumsum += size
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✓ Saved Manhattan plot data: {output_file}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge GWAS results from multiple jobs'
    )
    parser.add_argument(
        '-output_dir',
        type=str,
        required=True,
        help='Directory containing GWAS result CSV files'
    )
    parser.add_argument(
        '-p_threshold',
        type=float,
        default=5e-8,
        help='Genome-wide significance threshold (default: 5e-8)'
    )
    parser.add_argument(
        '-manhattan_data',
        action='store_true',
        help='Generate data for Manhattan plot visualization'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("GWAS Results Merger")
    print("="*60)
    print(f"Output directory: {args.output_dir}\n")
    
    try:
        all_merged, sig_merged, summary = merge_gwas_results(
            args.output_dir,
            p_threshold=args.p_threshold,
            verbose=True
        )
        
        if args.manhattan_data and all_merged is not None:
            print("\nGenerating Manhattan plot data...")
            manhattan_df = create_manhattan_plot_data(
                all_merged,
                os.path.join(args.output_dir, 'manhattan_plot_data.csv')
            )
        
        print("\nDone!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
