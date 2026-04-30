#!/usr/bin/env python3
"""
GWAS Disease-Specific SNP Filtering - Position-Only Matching with Complete Statistics

Matches on: chromosome + bp position (ignoring alleles due to UKBB ',' placeholders)
Tracks: Original UKBB SNPs, Original GWAS SNPs, Matched SNPs, P-value stats
Generates: Per-chromosome stats + Overall summary file
"""

import os
import pandas as pd
import numpy as np
import time
import argparse
import json
import gc
from datetime import datetime

BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8"
OUTPUT_BASE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/filtered_by_disease_gwas_summ_stats_5e8_threshold"

DISEASE_CONFIGS = {
    "pancreatic": {
        "name": "Pancreatic",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/model1_ukb_imp_chr1-22_panc_merged.txt",
        "column_mappings": {
            "chromosome": "CHR",
            "position": "BP",
            "allele1": "ALLELE1",
            "allele2": "ALLELE0",
            "pvalue": "P_LINREG"
        }
    },
    "t2d": {
        "name": "T2D",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/Mahajan.NatGenet2018b.T2D.European.txt",
        "column_mappings": {
            "chromosome": "Chr",
            "position": "Pos",
            "allele1": "EA",
            "allele2": "NEA",
            "pvalue": "Pvalue"
        }
    },
    "prostate": {
        "name": "Prostate",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/meta_v3_onco_euro_overall_ChrAll_1_release.txt",
        "column_mappings": {
            "chromosome": "Chr",
            "position": "position",
            "allele1": "Allele1",
            "allele2": "Allele2",
            "pvalue": "Pvalue"
        }
    },
    "colon": {
        "name": "Colon",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/joint_wald_noUKB_MAC50_1_rsID.TBL",
        "column_mappings": {
            "chromosome": "CHR",
            "position": "POS",
            "allele1": "Allele1",
            "allele2": "Allele2",
            "pvalue": "P.value"
        }
    },
    "breast": {
        "name": "Breast",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/bcac_meta_rs.txt",
        "column_mappings": {
            "chromosome": "chr",
            "position": "position_b37",
            "allele1": "a0",
            "allele2": "a1",
            "pvalue": "bcac_onco_icogs_gwas_P1df"
        }
    }
}

def process_chromosome_position_only(chr_num, disease_key, base_path, output_base,
                                     use_threshold, pvalue_threshold,
                                     output_formats=['npy', 'gen'], chunksize=500000):
    """
    Process using POSITION-ONLY matching (chr + bp)
    
    Returns dict with:
    - original_ukbb_snps: Total SNPs in UKBB file
    - original_gwas_snps: Total GWAS SNPs for this chr
    - matched_snps: Number of matched SNPs
    - pvalue_min, pvalue_max, pvalue_median
    - genome_wide_sig: Count of p < 5e-8
    """
    config = DISEASE_CONFIGS[disease_key]
    disease_name = config['name']
    gwas_file = config['gwas_file']
    col_map = config['column_mappings']
    
    chr_col = col_map['chromosome']
    pos_col = col_map['position']
    allele1_col = col_map['allele1']
    allele2_col = col_map['allele2']
    pval_col = col_map['pvalue']
    
    print(f"\n{'='*80}")
    print(f"Chr{chr_num} - {disease_name}")
    print(f"{'='*80}")
    print(f"  Matching strategy: Position-only (chr + bp)")
    print(f"  Ignoring alleles due to ',' placeholders in UKBB")    
    
    ukbb_file = os.path.join(base_path, f"chr{chr_num}", f"chr{chr_num}_variants.gen")
    
    if not os.path.exists(ukbb_file):
        print(f"  ERROR: UKBB file not found: {ukbb_file}")
        return {
            'chr': chr_num,
            'disease': disease_key,
            'success': False,
            'error': 'UKBB file not found'
        }
    
    # ===================================================================
    # STEP 1: Count GWAS SNPs before and after thresholding
    # ===================================================================
    print(f"  Counting GWAS SNPs for chr{chr_num}...")
    
    total_gwas_snps_before_threshold = 0
    total_gwas_snps_after_threshold = 0
    gwas_reader = pd.read_csv(gwas_file, sep=r'\s+', chunksize=chunksize, low_memory=False)
    
    for gwas_chunk in gwas_reader:
        # Filter by chromosome first
        gwas_chunk[chr_col] = gwas_chunk[chr_col].astype(str)
        gwas_chunk = gwas_chunk[gwas_chunk[chr_col] == str(chr_num)]
        
        # Count before threshold
        total_gwas_snps_before_threshold += len(gwas_chunk)
        
        # Apply p-value threshold if needed
        if use_threshold:
            gwas_chunk[pval_col] = pd.to_numeric(gwas_chunk[pval_col], errors='coerce')
            gwas_chunk = gwas_chunk[gwas_chunk[pval_col] < pvalue_threshold].copy()
        
        # Count after threshold
        total_gwas_snps_after_threshold += len(gwas_chunk)
    
    print(f"    GWAS SNPs (before threshold): {total_gwas_snps_before_threshold:,}")
    if use_threshold:
        filtered_out = total_gwas_snps_before_threshold - total_gwas_snps_after_threshold
        print(f"    GWAS SNPs (after p<{pvalue_threshold}): {total_gwas_snps_after_threshold:,}")
        print(f"    Filtered out by p-value: {filtered_out:,}")
    else:
        print(f"    (No p-value threshold applied)")
    
    # ===================================================================
    # STEP 2: Load UKBB file and match
    # ===================================================================
    print(f"  Loading UKBB chr{chr_num} and matching...")
    
    with open(ukbb_file, 'r') as f:
        first_line = f.readline().strip()
        has_header = any(keyword in first_line.lower() 
                        for keyword in ['rsid', 'position', 'chr', 'ref', 'alt'])
    
    ukbb_reader = pd.read_csv(
        ukbb_file,
        sep=r'\s+',
        header=0 if has_header else None,
        names=None if has_header else ['rsid', 'chromosome', 'bp', 'ref_allele', 'alt_allele'],
        chunksize=chunksize,
        low_memory=False
    )
    
    all_matches = []
    matched_indices = set()
    chunk_idx = 0
    total_ukbb_snps = 0
    
    for ukbb_chunk in ukbb_reader:
        chunk_idx += 1
        total_ukbb_snps += len(ukbb_chunk)
        
        # Standardize column names
        if has_header:
            col_rename = {}
            for col in ukbb_chunk.columns:
                col_lower = col.lower()
                if 'rsid' in col_lower or col_lower == 'snp':
                    col_rename[col] = 'rsid'
                elif col_lower == 'chr' or col_lower == 'chromosome':
                    col_rename[col] = 'chromosome'
                elif 'position' in col_lower or col_lower == 'bp':
                    col_rename[col] = 'bp'
                elif 'ref' in col_lower and 'allele' in col_lower:
                    col_rename[col] = 'ref_allele'
                elif 'alt' in col_lower and 'allele' in col_lower:
                    col_rename[col] = 'alt_allele'
            ukbb_chunk = ukbb_chunk.rename(columns=col_rename)
        
        # Add SNP_Index
        base_index = (chunk_idx - 1) * chunksize
        ukbb_chunk['SNP_Index'] = np.arange(base_index, base_index + len(ukbb_chunk))
        
        # Convert types
        ukbb_chunk['chromosome'] = ukbb_chunk['chromosome'].astype(str)
        ukbb_chunk['bp'] = pd.to_numeric(ukbb_chunk['bp'], errors='coerce')
        ukbb_chunk = ukbb_chunk.dropna(subset=['bp'])
        ukbb_chunk['bp'] = ukbb_chunk['bp'].astype(np.int32)
        
        # ===================================================================
        # STEP 2: Process GWAS file in chunks for matching
        # ===================================================================
        gwas_reader2 = pd.read_csv(gwas_file, sep=r'\s+', chunksize=chunksize, low_memory=False)
        
        for gwas_chunk in gwas_reader2:
            # Apply p-value threshold
            if use_threshold:
                gwas_chunk[pval_col] = pd.to_numeric(gwas_chunk[pval_col], errors='coerce')
                gwas_chunk = gwas_chunk[gwas_chunk[pval_col] < pvalue_threshold].copy()
                if len(gwas_chunk) == 0:
                    continue
            
            # Filter by chromosome
            gwas_chunk[chr_col] = gwas_chunk[chr_col].astype(str)
            gwas_chunk = gwas_chunk[gwas_chunk[chr_col] == str(chr_num)]
            if len(gwas_chunk) == 0:
                continue
            
            # Convert position
            gwas_chunk[pos_col] = pd.to_numeric(gwas_chunk[pos_col], errors='coerce')
            gwas_chunk = gwas_chunk.dropna(subset=[pos_col])
            gwas_chunk[pos_col] = gwas_chunk[pos_col].astype(np.int32)
            
            # Match on chromosome + position ONLY
            merged_df = pd.merge(
                ukbb_chunk,
                gwas_chunk,
                left_on=['chromosome', 'bp'],
                right_on=[chr_col, pos_col],
                how='inner'
            )
            
            if len(merged_df) > 0:
                # Remove duplicates by SNP_Index
                if 'SNP_Index' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['SNP_Index'], keep='first')
                
                matched_indices.update(merged_df['SNP_Index'].tolist())
                all_matches.append(merged_df)
            
            del gwas_chunk
            gc.collect()
        
        if chunk_idx % 2 == 0:
            print(f"    Processed {chunk_idx} UKBB chunks, {len(matched_indices):,} matches so far")
        
        del ukbb_chunk
        gc.collect()
    
    print(f"  UKBB SNPs: {total_ukbb_snps:,}")
    
    # ===================================================================
    # STEP 3: Consolidate and save results
    # ===================================================================
    
    if len(all_matches) == 0:
        print(f"  WARNING: No matches found")
        return {
            'chr': chr_num,
            'disease': disease_key,
            'disease_name': disease_name,
            'original_ukbb_snps': total_ukbb_snps,
            'original_gwas_snps_before_threshold': total_gwas_snps_before_threshold,
            'original_gwas_snps_after_threshold': total_gwas_snps_after_threshold,
            'matched_snps': 0,
            'retention_rate': 0.0,
            'pvalue_min': None,
            'pvalue_max': None,
            'pvalue_median': None,
            'genome_wide_sig': 0,
            'success': True
        }
    
    # Combine all matches
    final_matches = pd.concat(all_matches, ignore_index=True)
    final_matches = final_matches.drop_duplicates(subset=['SNP_Index'], keep='first')
    
    indices = np.sort(final_matches['SNP_Index'].values)
    
    print(f"  Matched SNPs: {len(indices):,}")
    retention_rate = 100 * len(indices) / total_ukbb_snps if total_ukbb_snps > 0 else 0
    print(f"  Retention rate: {retention_rate:.2f}%")
    
    # ===================================================================
    # STEP 4: Save results
    # ===================================================================
    disease_dir = os.path.join(output_base, disease_key, f"chr{chr_num}")
    os.makedirs(disease_dir, exist_ok=True)
    
    output_files = {}
    
    # Save indices
    if 'npy' in output_formats:
        indices_npy = os.path.join(disease_dir, f"chr{chr_num}_filtered_indices_{disease_key}.npy")
        np.save(indices_npy, indices)
        output_files['indices_npy'] = indices_npy
        print(f"    Saved: {indices_npy}")
    
    if 'txt' in output_formats:
        indices_txt = os.path.join(disease_dir, f"chr{chr_num}_filtered_indices_{disease_key}.txt")
        np.savetxt(indices_txt, indices, fmt='%d')
        output_files['indices_txt'] = indices_txt
    
    # Save filtered variants
    if 'gen' in output_formats:
        filtered_gen = os.path.join(disease_dir, f"chr{chr_num}_variants_filtered_{disease_key}.gen")
        
        output_df = final_matches.copy()
        if 'SNP_Index' in output_df.columns:
            output_df = output_df.rename(columns={'SNP_Index': 'original_idx'})
        
        output_df.to_csv(filtered_gen, sep='\t', index=False, header=True)
        output_files['filtered_gen'] = filtered_gen
        print(f"    Saved: {filtered_gen}")
    
    # Calculate statistics
    stats = {
        'chromosome': chr_num,
        'disease': disease_key,
        'disease_name': disease_name,
        'original_ukbb_snps': total_ukbb_snps,
        'original_gwas_snps_before_threshold': total_gwas_snps_before_threshold,
        'original_gwas_snps_after_threshold': total_gwas_snps_after_threshold,
        'gwas_snps_filtered_out': total_gwas_snps_before_threshold - total_gwas_snps_after_threshold,
        'matched_snps': len(indices),
        'retention_rate': retention_rate,
        'matching_strategy': 'position_only',
        'p_value_threshold': pvalue_threshold if use_threshold else None,
        #'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add p-value statistics
    if pval_col in final_matches.columns:
        pvalues = pd.to_numeric(final_matches[pval_col], errors='coerce').dropna()
        if len(pvalues) > 0:
            stats['pvalue_min'] = float(pvalues.min())
            stats['pvalue_max'] = float(pvalues.max())
            #stats['pvalue_median'] = float(pvalues.median())
            stats['genome_wide_sig'] = int((pvalues < 5e-8).sum())
            
            if stats['genome_wide_sig'] > 0:
                print(f"    Genome-wide significant SNPs: {stats['genome_wide_sig']:,}")
        else:
            stats['pvalue_min'] = None
            stats['pvalue_max'] = None
            #stats['pvalue_median'] = None
            stats['genome_wide_sig'] = 0
    else:
        stats['pvalue_min'] = None
        stats['pvalue_max'] = None
        #stats['pvalue_median'] = None
        stats['genome_wide_sig'] = 0
    
    # Save stats file
    stats_file = os.path.join(disease_dir, f"chr{chr_num}_filtering_stats_{disease_key}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    output_files['stats'] = stats_file
    
    stats['output_files'] = output_files
    stats['success'] = True
    
    return stats


def create_summary_files(all_stats, diseases, output_base, pvalue_threshold, use_threshold):
    """Create comprehensive summary files for each disease"""
    
    print("\n" + "="*80)
    print("CREATING SUMMARY FILES")
    print("="*80)
    
    for disease_key in diseases:
        disease_stats = [s for s in all_stats 
                        if s.get('disease') == disease_key and s.get('success')]
        if not disease_stats:
            continue
        
        disease_name = DISEASE_CONFIGS[disease_key]['name']
        disease_dir = os.path.join(output_base, disease_key)
        os.makedirs(disease_dir, exist_ok=True)
        
        print(f"\n  {disease_name}:")
        
        # Create per-chromosome summary data
        summary_rows = []
        for stats in sorted(disease_stats, key=lambda x: x.get('chr', 999)):
            summary_rows.append({
                'chromosome': stats.get('chr'),
                'original_ukbb_snps': stats.get('original_ukbb_snps', 0),
                'original_gwas_snps_before_threshold': stats.get('original_gwas_snps_before_threshold', 0),
                'original_gwas_snps_after_threshold': stats.get('original_gwas_snps_after_threshold', 0),
                'gwas_snps_filtered_out': stats.get('gwas_snps_filtered_out', 0),
                'matched_snps': stats.get('matched_snps', 0),
                'retention_rate': stats.get('retention_rate', 0),
                'pvalue_min': stats.get('pvalue_min'),
                'pvalue_max': stats.get('pvalue_max'),
                #'pvalue_median': stats.get('pvalue_median'),
                'genome_wide_sig': stats.get('genome_wide_sig', 0)
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Calculate totals
        total_ukbb = summary_df['original_ukbb_snps'].sum()
        total_gwas_before = summary_df['original_gwas_snps_before_threshold'].sum()
        total_gwas_after = summary_df['original_gwas_snps_after_threshold'].sum()
        total_gwas_filtered = summary_df['gwas_snps_filtered_out'].sum()
        total_matched = summary_df['matched_snps'].sum()
        total_retention = 100 * total_matched / total_ukbb if total_ukbb > 0 else 0
        
        # Get min/max pvalues across all chromosomes
        pvalue_mins = summary_df['pvalue_min'].dropna()
        pvalue_maxs = summary_df['pvalue_max'].dropna()
        #pvalue_medians = summary_df['pvalue_median'].dropna()
        
        totals = {
            'chromosome': 'TOTAL',
            'original_ukbb_snps': total_ukbb,
            'original_gwas_snps_before_threshold': total_gwas_before,
            'original_gwas_snps_after_threshold': total_gwas_after,
            'gwas_snps_filtered_out': total_gwas_filtered,
            'matched_snps': total_matched,
            'retention_rate': total_retention,
            'pvalue_min': float(pvalue_mins.min()) if len(pvalue_mins) > 0 else None,
            'pvalue_max': float(pvalue_maxs.max()) if len(pvalue_maxs) > 0 else None,
            #'pvalue_median': float(pvalue_medians.median()) if len(pvalue_medians) > 0 else None,
            'genome_wide_sig': summary_df['genome_wide_sig'].sum()
        }
        
        # Append totals row
        summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
        
        # Save as CSV
        summary_csv = os.path.join(disease_dir, f"{disease_key}_filtering_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"    Saved CSV: {summary_csv}")
        
        # Create detailed JSON summary
        summary_json_data = {
            'disease': disease_key,
            'disease_name': disease_name,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'chromosomes_processed': len(disease_stats),
            'matching_strategy': 'position_only',
            'p_value_threshold': pvalue_threshold if use_threshold else None,
            'per_chromosome': summary_rows,
            'totals': {
                'original_ukbb_snps': int(total_ukbb),
                'original_gwas_snps_before_threshold': int(total_gwas_before),
                'original_gwas_snps_after_threshold': int(total_gwas_after),
                'gwas_snps_filtered_out': int(total_gwas_filtered),
                'matched_snps': int(total_matched),
                'retention_rate': round(total_retention, 2),
                'pvalue_min': totals['pvalue_min'],
                'pvalue_max': totals['pvalue_max'],
                #'pvalue_median': totals['pvalue_median'],
                'genome_wide_sig': int(totals['genome_wide_sig'])
            }
        }
        
        summary_json = os.path.join(disease_dir, f"{disease_key}_filtering_summary.json")
        with open(summary_json, 'w') as f:
            json.dump(summary_json_data, f, indent=2)
        print(f"    Saved JSON: {summary_json}")
        
        # Print summary to console
        print(f"\n    Summary for {disease_name}:")
        print(f"      Total UKBB SNPs: {total_ukbb:,}")
        print(f"      Total GWAS SNPs (before threshold): {total_gwas_before:,}")
        print(f"      Total GWAS SNPs (after threshold): {total_gwas_after:,}")
        if total_gwas_filtered > 0:
            print(f"      GWAS SNPs filtered out: {total_gwas_filtered:,}")
        print(f"      Total Matched: {total_matched:,}")
        print(f"      Retention Rate: {total_retention:.2f}%")
        if totals['genome_wide_sig'] > 0:
            print(f"      Genome-wide Significant: {totals['genome_wide_sig']:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter UKBB SNPs by GWAS - Position-Only Matching with Statistics'
    )
    
    parser.add_argument('-base_path', type=str, default=BASE_PATH, help='Base path for UKBB chromosome data')
    parser.add_argument('-output_base', type=str, default=OUTPUT_BASE, help='Base path for output files')
    parser.add_argument('-diseases', type=str, default='all', help='Comma-separated disease keys (t2d,prostate,pancreatic,colon,breast) or "all"')
    parser.add_argument('-chromosomes', type=str, default='1-22', help='Chromosomes to process (e.g., "1-22" or "1,2,3")')
    parser.add_argument('-output_formats', type=str, default='npy,gen', help='Output formats (comma-separated): npy, txt, gen. Default: npy,gen')
    parser.add_argument('-use_threshold', action='store_false', help='Enable p-value thresholding')
    parser.add_argument('-threshold', type=float, default=5e-8, help='P-value threshold if -use_threshold is enabled (default: 0.1)')
    parser.add_argument('-chunksize', type=int, default=500000, help='Chunk size (default: 500000)')
    
    args = parser.parse_args()
    
    output_formats = [fmt.strip() for fmt in args.output_formats.split(',')]
    
    if args.diseases.lower() == 'all':
        diseases = list(DISEASE_CONFIGS.keys())
    else:
        diseases = [d.strip() for d in args.diseases.split(',')]
    
    if '-' in args.chromosomes:
        start, end = map(int, args.chromosomes.split('-'))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]
    
    print("\n" + "="*80)
    print("GWAS FILTERING - Position-Only Matching with Complete Statistics")
    print("="*80)
    print(f"\nMatching Strategy: chr + bp (IGNORING alleles)")
    print(f"Reason: UKBB alt_allele has ',' placeholders")
    print(f"\nConfiguration:")
    print(f"  UKBB path: {args.base_path}")
    print(f"  Output path: {args.output_base}")
    print(f"  Diseases: {', '.join([DISEASE_CONFIGS[d]['name'] for d in diseases])}")
    print(f"  Chromosomes: {chromosomes}")
    print(f"  P-value threshold: {args.threshold if args.use_threshold else 'None'}")
    print(f"  Chunk size: {args.chunksize:,}")
    
    all_stats = []
    total_start = time.time()
    
    for disease_key in diseases:
        disease_name = DISEASE_CONFIGS[disease_key]['name']
        gwas_file = DISEASE_CONFIGS[disease_key]['gwas_file']
        
        print("\n" + "="*80)
        print(f"DISEASE: {disease_name} ({disease_key})")
        print("="*80)
        print(f"GWAS file: {gwas_file}")
        
        if not os.path.exists(gwas_file):
            print(f"ERROR: GWAS file not found!")
            continue
        
        for chr_num in chromosomes:
            stats = process_chromosome_position_only(
                chr_num, disease_key,
                args.base_path, args.output_base,
                args.use_threshold, args.threshold,
                output_formats, args.chunksize
            )
            all_stats.append(stats)
            gc.collect()
    
    # Create summary files
    create_summary_files(all_stats, diseases, args.output_base, 
                        args.threshold, args.use_threshold)
    
    total_elapsed = time.time() - total_start
    
    # Console summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    successful = [s for s in all_stats if s.get('success', False)]
    
    if successful:
        print(f"\n✓ Successfully processed {len(successful)} chromosome-disease combinations\n")
        
        for disease_key in diseases:
            disease_stats = [s for s in successful if s.get('disease') == disease_key]
            if not disease_stats:
                continue
            
            disease_name = DISEASE_CONFIGS[disease_key]['name']
            print(f"{disease_name}:")
            print(f"  {'Chr':<5} {'UKBB SNPs':<12} {'GWAS Before':<12} {'GWAS After':<12} {'Matched':<12} {'%':<8}")
            print(f"  {'-'*70}")
            
            total_ukbb = 0
            total_gwas_before = 0
            total_gwas_after = 0
            total_matched = 0
            
            for s in sorted(disease_stats, key=lambda x: x.get('chr', 999)):
                ukbb = s.get('original_ukbb_snps', 0)
                gwas_before = s.get('original_gwas_snps_before_threshold', 0)
                gwas_after = s.get('original_gwas_snps_after_threshold', 0)
                matched = s.get('matched_snps', 0)
                
                total_ukbb += ukbb
                total_gwas_before += gwas_before
                total_gwas_after += gwas_after
                total_matched += matched
                
                rate = s.get('retention_rate', 0)
                print(f"  {s.get('chr'):<5} {ukbb:<12,} {gwas_before:<12,} {gwas_after:<12,} {matched:<12,} {rate:<8.2f}")
            
            print(f"  {'-'*70}")
            total_rate = 100 * total_matched / total_ukbb if total_ukbb > 0 else 0
            print(f"  {'Total':<5} {total_ukbb:<12,} {total_gwas_before:<12,} {total_gwas_after:<12,} {total_matched:<12,} {total_rate:<8.2f}\n")
    
    print(f"Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n✓ Complete! Summary files saved in each disease directory.\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())