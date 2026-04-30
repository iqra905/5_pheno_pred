#!/usr/bin/env python3
"""
Reverse SNP Filtering: Filter EPIC SNPs based on UKBB chromosome variants.

For each chromosome, finds SNPs common between the EPIC master file and
the UKBB chromosome variant file, then saves:
  - A CSV of common SNPs with their EPIC indices
  - A .npy file of EPIC indices for fast dataloader access

Usage:
    python epic_ukbb_common_snp_filtering.py
    python epic_ukbb_common_snp_filtering.py -chromosomes 1-5
    python epic_ukbb_common_snp_filtering.py -chromosomes 1,3,5
"""

import os
import pandas as pd
import numpy as np
import time
import argparse

# Configuration
EPIC_SNP_FILE = "/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen"
UKBB_BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/variants_metadata_chr"
OUTPUT_DIR    = "/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/meta_data/epic_ukbb_common_snps"


# =============================================================================
# HELPERS
# =============================================================================

def _detect_and_load(file_path, default_cols):
    """
    Auto-detect header and load a whitespace-separated SNP file.
    Returns a DataFrame with standardised column names.
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    has_header = any(kw in first_line.lower()
                     for kw in ['rsid', 'position', 'chr', 'ref', 'alt', 'allele'])

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=0 if has_header else None,
        names=None if has_header else default_cols,
        dtype=str
    )

    # Standardise column names when file has its own header
    if has_header:
        mapping = {}
        for col in df.columns:
            cl = col.lower()
            if cl in ('chr', 'chromosome'):
                mapping[col] = 'chr'
            elif 'rsid' in cl or cl in ('snp', 'id'):
                mapping[col] = 'rsid'
            elif 'position' in cl or cl in ('bp', 'pos'):
                mapping[col] = 'bp'
            elif 'ref' in cl and 'allele' in cl:
                mapping[col] = 'ref_allele'
            elif 'alt' in cl and 'allele' in cl:
                mapping[col] = 'alt_allele'
        df = df.rename(columns=mapping)

    df['bp'] = pd.to_numeric(df['bp'], errors='coerce').astype('Int64')
    return df


def load_epic_file(epic_file_path):
    """
    Load the full EPIC master SNP file (all chromosomes).
    Adds an 'epic_index' column — the row position in this file.
    """
    print(f"Loading EPIC SNP file: {epic_file_path}")
    t = time.time()

    df = _detect_and_load(
        epic_file_path,
        default_cols=['chr', 'rsid', 'bp', 'ref_allele', 'alt_allele']
    )
    df = df[['chr', 'rsid', 'bp', 'ref_allele', 'alt_allele']].copy()
    df['epic_index'] = np.arange(len(df))          # row index in the EPIC file

    print(f"  Loaded {len(df):,} EPIC SNPs in {time.time()-t:.2f}s")
    print(f"  Chromosomes present: {sorted(df['chr'].unique())}")
    return df


def load_ukbb_chromosome_variants(chr_num, base_path):
    """
    Load the UKBB variant metadata file for one chromosome.
    Located at: <base_path>/chr{N}/chr{N}_variants.gen
    """
    variant_file = os.path.join(base_path, f"chr{chr_num}", f"chr{chr_num}_variants.gen")
    if not os.path.exists(variant_file):
        raise FileNotFoundError(f"UKBB variant file not found: {variant_file}")

    print(f"  Loading UKBB chr{chr_num} variants: {variant_file}")
    df = _detect_and_load(
        variant_file,
        default_cols=['rsid', 'chr', 'bp', 'ref_allele', 'alt_allele']
    )
    df = df[['rsid', 'chr', 'bp', 'ref_allele', 'alt_allele']].copy()
    print(f"    {len(df):,} UKBB SNPs on chr{chr_num}")
    return df


def find_common_snps(chr_num, epic_snps, ukbb_variants):
    """
    Find SNPs present in both EPIC and UKBB for a given chromosome.
    Matching is done on rsid + bp position.

    Returns a DataFrame with columns:
        chr, rsid, bp, ref_allele, alt_allele, epic_index
    sorted by epic_index.
    """
    print(f"  Finding common SNPs for chr{chr_num}...")
    t = time.time()

    # Subset EPIC to this chromosome
    epic_chr = epic_snps[epic_snps['chr'] == str(chr_num)].copy()
    print(f"    EPIC SNPs on chr{chr_num}: {len(epic_chr):,}")

    # Create matching keys (rsid + bp) — same logic as original script
    epic_chr['match_key']     = epic_chr['rsid'].astype(str) + '_' + epic_chr['bp'].astype(str)
    ukbb_variants = ukbb_variants.copy()
    ukbb_variants['match_key'] = ukbb_variants['rsid'].astype(str) + '_' + ukbb_variants['bp'].astype(str)

    # Inner merge: EPIC (source) against UKBB (reference) — mirrors original
    merged = epic_chr.merge(
        ukbb_variants[['match_key']],
        on='match_key',
        how='inner'
    )

    # Sort by epic_index to preserve original EPIC order
    common = merged.drop(columns=['match_key']).sort_values('epic_index').reset_index(drop=True)

    retention = 100 * len(common) / len(epic_chr) if len(epic_chr) > 0 else 0
    print(f"    Common SNPs: {len(common):,} / {len(epic_chr):,} EPIC SNPs ({retention:.2f}%) "
          f"in {time.time()-t:.2f}s")

    return common


def save_results(chr_num, common_snps, output_dir):
    """
    Save outputs for one chromosome:
      - <output_dir>/epic_ukbb_common_snps_chr{N}.csv   — full SNP metadata + epic_index
      - <output_dir>/epic_indices_chr{N}.npy             — epic_index array for fast loading
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"epic_ukbb_common_snps_chr{chr_num}.csv")
    npy_path = os.path.join(output_dir, f"epic_indices_chr{chr_num}.npy")

    # CSV: all columns including epic_index
    common_snps.to_csv(csv_path, index=False)
    print(f"    Saved CSV:    {csv_path}  ({len(common_snps):,} rows)")

    # NPY: just the epic indices
    np.save(npy_path, common_snps['epic_index'].values.astype(np.int64))
    print(f"    Saved NPY:    {npy_path}  ({len(common_snps):,} indices)")

    return {'csv': csv_path, 'npy': npy_path}


# =============================================================================
# PER-CHROMOSOME PROCESSING
# =============================================================================

def process_chromosome(chr_num, epic_snps, base_path, output_dir):
    """Load UKBB variants, find common SNPs with EPIC, save results."""
    print(f"\n{'='*70}")
    print(f"Chromosome {chr_num}")
    print(f"{'='*70}")

    try:
        ukbb_variants = load_ukbb_chromosome_variants(chr_num, base_path)
        common_snps   = find_common_snps(chr_num, epic_snps, ukbb_variants)
        output_files  = save_results(chr_num, common_snps, output_dir)

        epic_on_chr = len(epic_snps[epic_snps['chr'] == str(chr_num)])
        return {
            'chr': chr_num,
            'epic_snps_on_chr': epic_on_chr,
            'ukbb_snps': len(ukbb_variants),
            'common_snps': len(common_snps),
            'retention_pct': 100 * len(common_snps) / epic_on_chr if epic_on_chr > 0 else 0,
            'output_files': output_files,
            'common_df': common_snps,   # retained for merged output
            'success': True
        }

    except Exception as e:
        import traceback
        print(f"\n  ERROR on chr{chr_num}: {e}")
        traceback.print_exc()
        return {'chr': chr_num, 'success': False, 'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find common SNPs between EPIC and UKBB; save CSV + NPY indices.'
    )
    parser.add_argument('-epic_file',   type=str, default=EPIC_SNP_FILE,
                        help='Path to EPIC master SNP file')
    parser.add_argument('-ukbb_path',   type=str, default=UKBB_BASE_PATH,
                        help='Base path containing chr{N}/chr{N}_variants.gen files')
    parser.add_argument('-output_dir',  type=str, default=OUTPUT_DIR,
                        help='Directory for output CSV and NPY files')
    parser.add_argument('-chromosomes', type=str, default='1-22',
                        help='Chromosomes to process: range "1-22" or list "1,2,3"')
    args = parser.parse_args()

    # Parse chromosome list
    if '-' in args.chromosomes and ',' not in args.chromosomes:
        start, end = map(int, args.chromosomes.split('-'))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]

    print("=" * 70)
    print("EPIC → UKBB COMMON SNP FILTERING")
    print("=" * 70)
    print(f"  EPIC file:    {args.epic_file}")
    print(f"  UKBB path:    {args.ukbb_path}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Chromosomes:  {chromosomes}")

    # Load EPIC once — reused for every chromosome
    print()
    epic_snps = load_epic_file(args.epic_file)

    # Process chromosomes sequentially
    all_stats       = []
    all_common_dfs  = []   # collect per-chromosome DataFrames for merged output
    total_start     = time.time()

    for chr_num in chromosomes:
        stats = process_chromosome(chr_num, epic_snps, args.ukbb_path, args.output_dir)
        all_stats.append(stats)
        if stats.get('success') and stats.get('common_df') is not None:
            all_common_dfs.append(stats['common_df'])

    # --- Merged output across all chromosomes ---
    if all_common_dfs:
        print(f"\n{'='*70}")
        print("Saving merged output (all chromosomes)")
        print(f"{'='*70}")

        merged_df = pd.concat(all_common_dfs, ignore_index=True)

        merged_csv = os.path.join(args.output_dir, "epic_ukbb_common_snps_all_chr.csv")
        merged_npy = os.path.join(args.output_dir, "epic_indices_all_chr.npy")

        merged_df.to_csv(merged_csv, index=False)
        print(f"  Saved merged CSV: {merged_csv}  ({len(merged_df):,} rows)")

        np.save(merged_npy, merged_df['epic_index'].values.astype(np.int64))
        print(f"  Saved merged NPY: {merged_npy}  ({len(merged_df):,} indices)")

    # Summary
    total_elapsed = time.time() - total_start
    successful    = [s for s in all_stats if s.get('success')]
    failed        = [s for s in all_stats if not s.get('success')]

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if successful:
        print(f"\n{'Chr':<6} {'EPIC SNPs':<14} {'UKBB SNPs':<14} "
              f"{'Common':<14} {'Retention %'}")
        print("-" * 65)
        total_epic = total_ukbb = total_common = 0
        for s in successful:
            total_epic   += s['epic_snps_on_chr']
            total_ukbb   += s['ukbb_snps']
            total_common += s['common_snps']
            print(f"{s['chr']:<6} {s['epic_snps_on_chr']:<14,} {s['ukbb_snps']:<14,} "
                  f"{s['common_snps']:<14,} {s['retention_pct']:.2f}%")
        print("-" * 65)
        overall_ret = 100 * total_common / total_epic if total_epic > 0 else 0
        print(f"{'Total':<6} {total_epic:<14,} {total_ukbb:<14,} "
              f"{total_common:<14,} {overall_ret:.2f}%")

    if failed:
        print(f"\nFailed chromosomes:")
        for s in failed:
            print(f"  Chr {s['chr']}: {s.get('error', 'unknown error')}")

    print(f"\nTotal time: {total_elapsed:.2f}s")
    print(f"\nPer-chromosome files in: {args.output_dir}")
    print("  epic_ukbb_common_snps_chr{{N}}.csv  — per-chr SNP metadata + epic_index")
    print("  epic_indices_chr{{N}}.npy           — per-chr EPIC indices")
    print(f"\nMerged files (all chromosomes):")
    print("  epic_ukbb_common_snps_all_chr.csv  — all common SNPs across all chr")
    print("  epic_indices_all_chr.npy           — all EPIC indices across all chr")

    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
