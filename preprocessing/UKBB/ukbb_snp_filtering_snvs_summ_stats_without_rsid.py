#!/usr/bin/env python3
"""
UKBB Two-Stage SNP Filtering Pipeline

Stage 1 - SNV Filtering:
    Keeps only single nucleotide variants (SNVs) where both ref and alt alleles
    have length == 1. Removes indels, multiallelic sites, and missing/invalid alleles.

Stage 2 - GWAS Summary Stats Filtering:
    Matches SNV-filtered variants against disease-specific GWAS summary statistics
    using chromosome + base-pair position matching (alleles intentionally ignored
    due to UKBB ',' placeholders). Optionally applies a p-value threshold.

Index convention:
    All saved indices reference positions in the ORIGINAL UKBB .gen file,
    so downstream dataloaders can use them directly without any remapping.

Outputs (per chromosome, per disease):
    - chr{N}_snvs_summ_stats_filtered_indices_{disease}.npy   <- original-file indices
    - chr{N}_variants_snvs_summ_stats_filtered_{disease}.gen  <- matched variant rows
    - chr{N}_two_stage_filtering_stats_{disease}.json   <- per-chr statistics

Outputs (per disease):
    - {disease}_two_stage_filtering_summary.csv
    - {disease}_two_stage_filtering_summary.json

Outputs (SNV stage, per chromosome):
    - chr{N}_snv_filtered_indices.npy
    - chr{N}_variants_snv_only.gen

Global summary:
    - snv_filtering_summary.csv / .json
"""

import os
import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT PATHS
# =============================================================================

BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/variants_metadata_chr"
OUTPUT_BASE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/snvs_summ_stats_filtered_0.005_without_rsid"


# =============================================================================
# DISEASE CONFIGURATIONS  (from script 1)
# =============================================================================

DISEASE_CONFIGS = {
    "pancreatic": {
        "name": "Pancreatic",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/model1_ukb_imp_chr1-22_panc_merged.txt",
        "column_mappings": {
            "chromosome": "CHR",
            "position":   "BP",
            "snp_id":     "SNP",       # rsid column — set to None if not present
            "allele1":    "ALLELE1",
            "allele2":    "ALLELE0",
            "pvalue":     "P_LINREG"
        }
    },
    "t2d": {
        "name": "T2D",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/Mahajan.NatGenet2018b.T2D.European.txt",
        "column_mappings": {
            "chromosome": "Chr",
            "position":   "Pos",
            "snp_id":     "SNP",
            "allele1":    "EA",
            "allele2":    "NEA",
            "pvalue":     "Pvalue"
        }
    },
    "prostate": {
        "name": "Prostate",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/meta_v3_onco_euro_overall_ChrAll_1_release.txt",
        "column_mappings": {
            "chromosome": "Chr",
            "position":   "position",
            "snp_id":     "SNP",
            "allele1":    "Allele1",
            "allele2":    "Allele2",
            "pvalue":     "Pvalue"
        }
    },
    "colon": {
        "name": "Colon",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/joint_wald_noUKB_MAC50_1_rsID.TBL",
        "column_mappings": {
            "chromosome": "CHR",
            "position":   "POS",
            "snp_id":     "V2",
            "allele1":    "Allele1",
            "allele2":    "Allele2",
            "pvalue":     "P.value"
        }
    },
    "breast": {
        "name": "Breast",
        "gwas_file": "/mnt/fast/datasets/ucdatasets/gwas/data_files/5D_snp_info_files/bcac_meta_rs.txt",
        "column_mappings": {
            "chromosome": "chr",
            "position":   "position_b37",
            "snp_id":     "rsid",
            "allele1":    "a0",
            "allele2":    "a1",
            "pvalue":     "bcac_onco_icogs_gwas_P1df"
        }
    }
}


# =============================================================================
# HELPERS
# =============================================================================

def _load_ukbb_variant_file(ukbb_file, chunksize=None):
    """
    Load a UKBB .gen variant metadata file, auto-detecting headers.

    Returns a DataFrame (or an iterator of chunks if chunksize is given)
    with standardised columns: rsid, chromosome, bp, ref_allele, alt_allele.
    """
    with open(ukbb_file, 'r') as fh:
        first_line = fh.readline().strip()
    has_header = any(kw in first_line.lower()
                     for kw in ['rsid', 'position', 'chr', 'ref', 'alt', 'allele'])

    reader = pd.read_csv(
        ukbb_file,
        sep=r'\s+',
        header=0 if has_header else None,
        names=None if has_header else ['rsid', 'chromosome', 'bp', 'ref_allele', 'alt_allele'],
        dtype=str,
        chunksize=chunksize,
        low_memory=False
    )

    def _rename(df):
        rename_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'rsid' in cl or cl == 'snp' or cl == 'id':
                rename_map[col] = 'rsid'
            elif cl in ('chr', 'chromosome'):
                rename_map[col] = 'chromosome'
            elif 'position' in cl or cl in ('bp', 'pos'):
                rename_map[col] = 'bp'
            elif 'ref' in cl and 'allele' in cl:
                rename_map[col] = 'ref_allele'
            elif 'alt' in cl and 'allele' in cl:
                rename_map[col] = 'alt_allele'
        return df.rename(columns=rename_map)

    if chunksize is None:
        return _rename(reader)          # single DataFrame
    else:
        # Return a generator that renames each chunk
        def _gen():
            for chunk in reader:
                yield _rename(chunk)
        return _gen()


# =============================================================================
# STAGE 1 – SNV FILTERING
# =============================================================================

def stage1_snv_filter(chr_num, base_path, snv_output_dir, output_formats):
    """
    Load chromosome variant file and retain only SNVs.

    Returns:
        snv_variants  : DataFrame of SNV-only rows (with reset index 0..n_snv-1)
        snv_indices   : np.ndarray of positions in the ORIGINAL file (int64)
        variant_types : dict of filtering statistics
    """
    ukbb_file = os.path.join(base_path, f"chr{chr_num}", f"chr{chr_num}_variants.gen")
    if not os.path.exists(ukbb_file):
        raise FileNotFoundError(f"UKBB file not found: {ukbb_file}")

    print(f"  [Stage 1] Loading UKBB chr{chr_num} variant file...")
    chr_variants = _load_ukbb_variant_file(ukbb_file)

    # Ensure bp is numeric
    chr_variants['bp'] = pd.to_numeric(chr_variants['bp'], errors='coerce')

    total_snps = len(chr_variants)
    print(f"    Loaded {total_snps:,} total variants")

    # ---- SNV mask ----
    mask = (
        (chr_variants['ref_allele'].str.len() == 1) &
        (chr_variants['alt_allele'].str.len() == 1) &
        ~chr_variants['ref_allele'].str.contains(',', na=False) &
        ~chr_variants['alt_allele'].str.contains(',', na=False) &
        chr_variants['ref_allele'].notna() &
        chr_variants['alt_allele'].notna() &
        (chr_variants['ref_allele'] != '') &
        (chr_variants['alt_allele'] != '')
    )

    snv_indices = np.sort(np.where(mask)[0]).astype(np.int64)
    snv_variants = chr_variants.iloc[snv_indices].reset_index(drop=True)

    snv_count = len(snv_indices)
    retention_pct = 100.0 * snv_count / total_snps if total_snps > 0 else 0.0

    # Breakdown of removed variants
    multi_allelic = (
        chr_variants['ref_allele'].str.contains(',', na=False) |
        chr_variants['alt_allele'].str.contains(',', na=False)
    ).sum()
    indel_mask = (
        (chr_variants['ref_allele'].str.len() > 1) |
        (chr_variants['alt_allele'].str.len() > 1)
    )
    # Avoid double-counting multiallelic indels
    indels = int(indel_mask.sum()) - int(
        (indel_mask & chr_variants['ref_allele'].str.contains(',', na=False)).sum()
    )
    invalid = int(
        (chr_variants['ref_allele'].isna() | chr_variants['alt_allele'].isna() |
         (chr_variants['ref_allele'] == '') | (chr_variants['alt_allele'] == '')).sum()
    )

    variant_types = {
        'total':                  total_snps,
        'snv_kept':               snv_count,
        'removed':                total_snps - snv_count,
        'retention_pct':          retention_pct,
        'indels_removed':         indels,
        'multiallelic_removed':   int(multi_allelic),
        'invalid_removed':        invalid,
    }

    print(f"    SNV filter: kept {snv_count:,} / {total_snps:,} ({retention_pct:.2f}%)")
    print(f"      Removed — indels: {indels:,}  multiallelic: {int(multi_allelic):,}  invalid: {invalid:,}")

    # ---- Save Stage-1 outputs ----
    chr_snv_dir = os.path.join(snv_output_dir, f"chr{chr_num}")
    os.makedirs(chr_snv_dir, exist_ok=True)

    if 'npy' in output_formats:
        npy_path = os.path.join(chr_snv_dir, f"chr{chr_num}_snv_filtered_indices.npy")
        np.save(npy_path, snv_indices)
        print(f"    Saved SNV indices: {npy_path}")

    if 'txt' in output_formats:
        txt_path = os.path.join(chr_snv_dir, f"chr{chr_num}_snv_filtered_indices.txt")
        np.savetxt(txt_path, snv_indices, fmt='%d')
        print(f"    Saved SNV indices (txt): {txt_path}")

    if 'gen' in output_formats:
        gen_path = os.path.join(chr_snv_dir, f"chr{chr_num}_variants_snv_only.gen")
        snv_variants.to_csv(gen_path, sep='\t', index=False)
        print(f"    Saved SNV-only .gen: {gen_path}")

    return snv_variants, snv_indices, variant_types


# =============================================================================
# STAGE 2 – GWAS SUMMARY STATS FILTERING  (operates on SNV-filtered variants)
# =============================================================================

def stage2_gwas_filter(chr_num, disease_key,
                       snv_variants, snv_indices,
                       gwas_output_dir,
                       use_threshold, pvalue_threshold,
                       output_formats, chunksize=500_000):
    """
    Match SNV-filtered variants for chr_num against disease GWAS summary stats
    using a two-tier matching strategy:

      Tier 2 — chr + bp + alleles      (forward AND reverse strand)
      Tier 3 — chr + bp position only  (fallback for anything unmatched above)

    The indices saved are in ORIGINAL UKBB file coordinates
    (i.e.  snv_indices[position_in_snv_df]).

    Returns a stats dict.
    """
    config       = DISEASE_CONFIGS[disease_key]
    disease_name = config['name']
    gwas_file    = config['gwas_file']
    col_map      = config['column_mappings']

    chr_col     = col_map['chromosome']
    pos_col     = col_map['position']
    pval_col    = col_map['pvalue']
    allele1_col = col_map['allele1']
    allele2_col = col_map['allele2']

    print(f"\n  [Stage 2] {disease_name} — chr{chr_num}")
    print(f"    Matching strategy: TIERED")
    print(f"      Tier 2: chr + bp + alleles        (fwd + rev strand)")
    print(f"      Tier 3: chr + bp position-only    (fallback)")

    if not os.path.exists(gwas_file):
        print(f"    ERROR: GWAS file not found: {gwas_file}")
        return {
            'chr': chr_num, 'disease': disease_key, 'disease_name': disease_name,
            'success': False, 'error': 'GWAS file not found'
        }

    # ------------------------------------------------------------------
    # Build tiered lookup dicts from SNV-filtered variants.
    #
    # After Stage 1 all alleles are guaranteed to be single characters
    # with no commas, so allele-level matching (Tier 2) is reliable.
    #
    # All dicts map  key -> snv_row_position (int, 0-based in snv_variants)
    # First-encounter wins for any duplicate positions.
    # ------------------------------------------------------------------
    snv_bp    = pd.to_numeric(snv_variants['bp'],         errors='coerce').fillna(-1).astype(np.int32)
    snv_chr   = snv_variants['chromosome'].astype(str)
    snv_ref   = snv_variants['ref_allele'].astype(str)
    snv_alt   = snv_variants['alt_allele'].astype(str)

    # Tier 2: (chr, bp, ref, alt)  — forward strand
    allele_fwd_lookup = {}
    # Tier 2: (chr, bp, alt, ref)  — reverse strand (GWAS allele order swapped)
    allele_rev_lookup = {}
    # Tier 3: (chr, bp)
    pos_lookup     = {}

    for row_pos, (ch, bp, ref, alt) in enumerate(
            zip(snv_chr, snv_bp, snv_ref, snv_alt)):
        ch  = str(ch)
        bp  = int(bp)
        ref = str(ref).upper()
        alt = str(alt).upper()

        # Tier 3 — always populate; keep first encounter
        if (ch, bp) not in pos_lookup:
            pos_lookup[(ch, bp)] = row_pos

        # Tier 2 — allele lookups (forward and reverse)
        if (ch, bp, ref, alt) not in allele_fwd_lookup:
            allele_fwd_lookup[(ch, bp, ref, alt)] = row_pos
        if (ch, bp, alt, ref) not in allele_rev_lookup:
            allele_rev_lookup[(ch, bp, alt, ref)] = row_pos

    print(f"    SNV lookup built — "
          f"Tier2-fwd: {len(allele_fwd_lookup):,}  "
          f"Tier3: {len(pos_lookup):,} entries")

    # ------------------------------------------------------------------
    # Stream GWAS file in chunks; count before/after threshold; match
    # ------------------------------------------------------------------
    total_gwas_before = 0
    total_gwas_after  = 0
    matched_snv_positions = set()   # snv row-positions already claimed
    matched_gwas_rows     = []      # (snv_pos, gwas_row) for output
    tier_counts = {2: 0, 3: 0}

    gwas_reader = pd.read_csv(gwas_file, sep=r'\s+', chunksize=chunksize, low_memory=False)

    for gwas_chunk in gwas_reader:
        # Chromosome filter
        gwas_chunk[chr_col] = gwas_chunk[chr_col].astype(str)
        gwas_chunk = gwas_chunk[gwas_chunk[chr_col] == str(chr_num)]
        if len(gwas_chunk) == 0:
            continue

        total_gwas_before += len(gwas_chunk)

        # P-value threshold
        if use_threshold:
            gwas_chunk[pval_col] = pd.to_numeric(gwas_chunk[pval_col], errors='coerce')
            gwas_chunk = gwas_chunk[gwas_chunk[pval_col] < pvalue_threshold].copy()

        total_gwas_after += len(gwas_chunk)
        if len(gwas_chunk) == 0:
            continue

        # Type conversions
        gwas_chunk[pos_col] = pd.to_numeric(gwas_chunk[pos_col], errors='coerce')
        gwas_chunk = gwas_chunk.dropna(subset=[pos_col])
        gwas_chunk[pos_col] = gwas_chunk[pos_col].astype(np.int32)

        has_allele_cols = (allele1_col in gwas_chunk.columns and
                           allele2_col in gwas_chunk.columns)

        # ---- Per-row tiered matching ----
        for _, gwas_row in gwas_chunk.iterrows():
            ch  = str(gwas_row[chr_col])
            bp  = int(gwas_row[pos_col])
            snv_pos = None
            tier_used = None

            # --- Tier 2: chr + bp + alleles (forward then reverse) ---
            if has_allele_cols:
                a1 = str(gwas_row[allele1_col]).upper()
                a2 = str(gwas_row[allele2_col]).upper()
                snv_pos = allele_fwd_lookup.get((ch, bp, a1, a2))
                if snv_pos is not None:
                    tier_used = 2
                else:
                    snv_pos = allele_rev_lookup.get((ch, bp, a1, a2))
                    if snv_pos is not None:
                        tier_used = 2

            # --- Tier 3: position only ---
            if snv_pos is None:
                snv_pos = pos_lookup.get((ch, bp))
                if snv_pos is not None:
                    tier_used = 3

            # Record match (first match per SNV position wins)
            if snv_pos is not None and snv_pos not in matched_snv_positions:
                matched_snv_positions.add(snv_pos)
                matched_gwas_rows.append((snv_pos, gwas_row, tier_used))
                tier_counts[tier_used] += 1

        del gwas_chunk
        gc.collect()

    gwas_filtered_out = total_gwas_before - total_gwas_after

    print(f"    GWAS SNPs on chr{chr_num} — before threshold: {total_gwas_before:,}")
    if use_threshold:
        print(f"    GWAS SNPs after p<{pvalue_threshold}: {total_gwas_after:,}  "
              f"(filtered out: {gwas_filtered_out:,})")

    # ------------------------------------------------------------------
    # Build final result
    # ------------------------------------------------------------------
    n_matched = len(matched_snv_positions)
    n_snv     = len(snv_variants)
    retention_rate = 100.0 * n_matched / n_snv if n_snv > 0 else 0.0

    print(f"    Matched SNPs (vs SNV set): {n_matched:,} / {n_snv:,} ({retention_rate:.2f}%)")
    print(f"    Matches by tier — "
          f"Tier2 (alleles): {tier_counts[2]:,}  "
          f"Tier3 (pos-only): {tier_counts[3]:,}")

    stats = {
        'chr':                                chr_num,
        'disease':                            disease_key,
        'disease_name':                       disease_name,
        'original_ukbb_snps':                 n_snv,        # after stage 1
        'original_gwas_snps_before_threshold': total_gwas_before,
        'original_gwas_snps_after_threshold':  total_gwas_after,
        'gwas_snps_filtered_out':             gwas_filtered_out,
        'matched_snps':                       n_matched,
        'retention_rate':                     retention_rate,
        'matching_strategy':                  'tiered_allele_position',
        'tier1_rsid_matches':                 0,
        'tier2_allele_matches':               tier_counts[2],
        'tier3_position_matches':             tier_counts[3],
        'p_value_threshold':                  pvalue_threshold if use_threshold else None,
        'pvalue_min':    None,
        'pvalue_max':    None,
        'genome_wide_sig': 0,
        'success':       True,
    }

    if n_matched == 0:
        print(f"    WARNING: No matches found for {disease_name} chr{chr_num}")
        return stats

    # Map matched snv-row positions → original UKBB indices
    matched_snv_positions_sorted = np.array(sorted(matched_snv_positions), dtype=np.int64)
    original_indices = np.sort(snv_indices[matched_snv_positions_sorted])

    # P-value statistics from matched rows
    pvalues = []
    for snv_pos, gwas_row, _tier in matched_gwas_rows:
        try:
            pvalues.append(float(gwas_row[pval_col]))
        except (ValueError, TypeError):
            pass

    if pvalues:
        pv_arr = np.array(pvalues)
        stats['pvalue_min']      = float(np.nanmin(pv_arr))
        stats['pvalue_max']      = float(np.nanmax(pv_arr))
        stats['genome_wide_sig'] = int((pv_arr < 5e-8).sum())
        if stats['genome_wide_sig'] > 0:
            print(f"    Genome-wide significant SNPs: {stats['genome_wide_sig']:,}")

    # ------------------------------------------------------------------
    # Save Stage-2 outputs
    # ------------------------------------------------------------------
    disease_chr_dir = os.path.join(gwas_output_dir, disease_key, f"chr{chr_num}")
    os.makedirs(disease_chr_dir, exist_ok=True)

    output_files = {}

    if 'npy' in output_formats:
        npy_path = os.path.join(disease_chr_dir,
                                f"chr{chr_num}_snvs_summ_stats_filtered_indices_{disease_key}.npy")
        np.save(npy_path, original_indices)
        output_files['indices_npy'] = npy_path
        print(f"    Saved indices (original coords): {npy_path}")

    if 'txt' in output_formats:
        txt_path = os.path.join(disease_chr_dir,
                                f"chr{chr_num}_snvs_summ_stats_filtered_indices_{disease_key}.txt")
        np.savetxt(txt_path, original_indices, fmt='%d')
        output_files['indices_txt'] = txt_path

    if 'gen' in output_formats:
        matched_snv_df = snv_variants.iloc[matched_snv_positions_sorted].copy()
        matched_snv_df.insert(0, 'original_idx', original_indices)

        # Append the matching GWAS columns (p-value etc.) as extra columns
        gwas_extras = []
        tier_labels = []
        for snv_pos, gwas_row, tier in sorted(matched_gwas_rows, key=lambda x: x[0]):
            gwas_extras.append(gwas_row)
            tier_labels.append(tier)
        if gwas_extras:
            gwas_extra_df = pd.DataFrame(gwas_extras).reset_index(drop=True)
            gwas_extra_df['match_tier'] = tier_labels
            # Drop columns already present in SNV df to avoid duplication
            overlap = set(matched_snv_df.columns) & set(gwas_extra_df.columns) - {'original_idx'}
            gwas_extra_df = gwas_extra_df.drop(columns=list(overlap), errors='ignore')
            matched_snv_df = pd.concat(
                [matched_snv_df.reset_index(drop=True), gwas_extra_df], axis=1
            )

        gen_path = os.path.join(disease_chr_dir,
                                f"chr{chr_num}_variants_snvs_summ_stats_filtered_{disease_key}.gen")
        matched_snv_df.to_csv(gen_path, sep='\t', index=False)
        output_files['filtered_gen'] = gen_path
        print(f"    Saved filtered .gen: {gen_path}")

    # Per-chr JSON stats
    stats_file = os.path.join(disease_chr_dir,
                              f"chr{chr_num}_two_stage_filtering_stats_{disease_key}.json")
    with open(stats_file, 'w') as fh:
        json.dump(stats, fh, indent=2, default=str)
    output_files['stats'] = stats_file
    stats['output_files'] = output_files

    return stats


# =============================================================================
# SUMMARY FILE CREATION
# =============================================================================

def create_snv_summary(snv_stats_list, snv_output_dir):
    """Save SNV-stage summary CSV and JSON."""
    print("\n  Creating SNV-stage summary files...")
    successful = [s for s in snv_stats_list if s.get('success')]
    if not successful:
        return

    df = pd.DataFrame(successful)
    cols = ['chr', 'total_snps', 'snv_kept', 'removed', 'retention_pct',
            'indels_removed', 'multiallelic_removed', 'invalid_removed']
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(snv_output_dir, "snv_filtering_summary.csv"), index=False)

    json_data = {
        'description':        'Stage 1 — SNV filtering results',
        'processing_date':    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filter_criteria':    {'ref_allele_length': 1, 'alt_allele_length': 1,
                               'no_commas': True, 'no_missing': True},
        'chromosomes_processed': len(successful),
        'totals': {
            'total_variants':     int(df['total_snps'].sum()),
            'snv_kept':           int(df['snv_kept'].sum()),
            'removed':            int(df['removed'].sum()),
            'retention_pct':      round(100.0 * df['snv_kept'].sum() /
                                        df['total_snps'].sum(), 2)
                                  if df['total_snps'].sum() > 0 else 0.0,
            'indels_removed':     int(df['indels_removed'].sum()),
            'multiallelic_removed': int(df['multiallelic_removed'].sum()),
        },
        'per_chromosome': df.to_dict(orient='records')
    }
    with open(os.path.join(snv_output_dir, "snv_filtering_summary.json"), 'w') as fh:
        json.dump(json_data, fh, indent=2, default=str)

    print(f"    Saved: {os.path.join(snv_output_dir, 'snv_filtering_summary.csv')}")
    print(f"    Saved: {os.path.join(snv_output_dir, 'snv_filtering_summary.json')}")


def create_gwas_summary(all_gwas_stats, diseases, gwas_output_dir,
                        pvalue_threshold, use_threshold):
    """Save GWAS-stage per-disease summary CSV and JSON files."""
    print("\n" + "=" * 80)
    print("CREATING GWAS-STAGE SUMMARY FILES")
    print("=" * 80)

    for disease_key in diseases:
        disease_stats = [s for s in all_gwas_stats
                         if s.get('disease') == disease_key and s.get('success')]
        if not disease_stats:
            continue

        disease_name = DISEASE_CONFIGS[disease_key]['name']
        disease_dir  = os.path.join(gwas_output_dir, disease_key)
        os.makedirs(disease_dir, exist_ok=True)

        print(f"\n  {disease_name}:")

        rows = []
        for s in sorted(disease_stats, key=lambda x: x.get('chr', 999)):
            rows.append({
                'chromosome':                          s.get('chr'),
                'snv_filtered_ukbb_snps':              s.get('original_ukbb_snps', 0),
                'original_gwas_snps_before_threshold': s.get('original_gwas_snps_before_threshold', 0),
                'original_gwas_snps_after_threshold':  s.get('original_gwas_snps_after_threshold', 0),
                'gwas_snps_filtered_out':              s.get('gwas_snps_filtered_out', 0),
                'matched_snps':                        s.get('matched_snps', 0),
                'retention_rate':                      s.get('retention_rate', 0),
                'tier1_rsid_matches':                  s.get('tier1_rsid_matches', 0),
                'tier2_allele_matches':                s.get('tier2_allele_matches', 0),
                'tier3_position_matches':              s.get('tier3_position_matches', 0),
                'pvalue_min':                          s.get('pvalue_min'),
                'pvalue_max':                          s.get('pvalue_max'),
                'genome_wide_sig':                     s.get('genome_wide_sig', 0),
            })

        summary_df = pd.DataFrame(rows)

        # Totals row
        total_snv     = summary_df['snv_filtered_ukbb_snps'].sum()
        total_before  = summary_df['original_gwas_snps_before_threshold'].sum()
        total_after   = summary_df['original_gwas_snps_after_threshold'].sum()
        total_filtered= summary_df['gwas_snps_filtered_out'].sum()
        total_matched = summary_df['matched_snps'].sum()
        total_ret     = 100.0 * total_matched / total_snv if total_snv > 0 else 0.0
        total_tier1   = summary_df['tier1_rsid_matches'].sum()
        total_tier2   = summary_df['tier2_allele_matches'].sum()
        total_tier3   = summary_df['tier3_position_matches'].sum()
        pv_mins = summary_df['pvalue_min'].dropna()
        pv_maxs = summary_df['pvalue_max'].dropna()

        totals_row = {
            'chromosome':                          'TOTAL',
            'snv_filtered_ukbb_snps':              total_snv,
            'original_gwas_snps_before_threshold': total_before,
            'original_gwas_snps_after_threshold':  total_after,
            'gwas_snps_filtered_out':              total_filtered,
            'matched_snps':                        total_matched,
            'retention_rate':                      total_ret,
            'tier1_rsid_matches':                  total_tier1,
            'tier2_allele_matches':                total_tier2,
            'tier3_position_matches':              total_tier3,
            'pvalue_min':                          float(pv_mins.min()) if len(pv_mins) else None,
            'pvalue_max':                          float(pv_maxs.max()) if len(pv_maxs) else None,
            'genome_wide_sig':                     summary_df['genome_wide_sig'].sum(),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([totals_row])], ignore_index=True)

        csv_path = os.path.join(disease_dir, f"{disease_key}_two_stage_filtering_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"    Saved CSV: {csv_path}")

        json_data = {
            'disease':                  disease_key,
            'disease_name':             disease_name,
            'processing_date':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pipeline':                 'two_stage_snv_then_gwas',
            'matching_strategy':        'tiered_allele_position',
            'p_value_threshold':        pvalue_threshold if use_threshold else None,
            'chromosomes_processed':    len(disease_stats),
            'per_chromosome':           rows,
            'totals': {
                'snv_filtered_ukbb_snps':              int(total_snv),
                'original_gwas_snps_before_threshold': int(total_before),
                'original_gwas_snps_after_threshold':  int(total_after),
                'gwas_snps_filtered_out':              int(total_filtered),
                'matched_snps':                        int(total_matched),
                'retention_rate':                      round(total_ret, 2),
                'tier1_rsid_matches':                  int(total_tier1),
                'tier2_allele_matches':                int(total_tier2),
                'tier3_position_matches':              int(total_tier3),
                'pvalue_min':                          totals_row['pvalue_min'],
                'pvalue_max':                          totals_row['pvalue_max'],
                'genome_wide_sig':                     int(totals_row['genome_wide_sig']),
            }
        }
        json_path = os.path.join(disease_dir, f"{disease_key}_two_stage_filtering_summary.json")
        with open(json_path, 'w') as fh:
            json.dump(json_data, fh, indent=2, default=str)
        print(f"    Saved JSON: {json_path}")

        print(f"\n    {disease_name} totals:")
        print(f"      SNV-filtered UKBB SNPs:          {total_snv:,}")
        print(f"      GWAS SNPs before threshold:      {total_before:,}")
        print(f"      GWAS SNPs after threshold:       {total_after:,}")
        print(f"      Final matched SNPs:              {total_matched:,}")
        print(f"      Retention rate (vs SNV set):     {total_ret:.2f}%")
        print(f"      Match breakdown by tier:")
        print(f"        Tier 2 (alleles):  {total_tier2:,}")
        print(f"        Tier 3 (pos-only): {total_tier3:,}")
        if totals_row['genome_wide_sig']:
            print(f"      Genome-wide significant:         {totals_row['genome_wide_sig']:,}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UKBB Two-Stage SNP Filtering: SNV filter → GWAS summary stats filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline:
  Stage 1 — SNV filter  : keep only variants with len(ref)==1 and len(alt)==1
  Stage 2 — GWAS filter : match SNV-filtered variants against disease GWAS summ stats

Index convention:
  All saved .npy index files reference positions in the ORIGINAL UKBB .gen file.

Examples:
  # Run full pipeline for all diseases, chromosomes 1-22
  %(prog)s

  # Only T2D and breast cancer, chromosomes 1-5, no p-value threshold
  %(prog)s -diseases t2d,breast -chromosomes 1-5 -use_threshold

  # Custom paths, save all formats
  %(prog)s -base_path /data/ukbb -output_base /results -output_formats npy,txt,gen
        """
    )

    parser.add_argument('-base_path',      type=str, default=BASE_PATH,
                        help='Base path for per-chromosome UKBB variant files')
    parser.add_argument('-output_base',    type=str, default=OUTPUT_BASE,
                        help='Root output directory')
    parser.add_argument('-diseases',       type=str, default='all',
                        help='Comma-separated disease keys or "all" '
                             '(choices: t2d, prostate, pancreatic, colon, breast)')
    parser.add_argument('-chromosomes',    type=str, default='1-22',
                        help='Chromosomes to process: "1-22" or "1,2,3"')
    parser.add_argument('-output_formats', type=str, default='npy,gen',
                        help='Output formats (comma-separated): npy, txt, gen. Default: npy,gen')
    parser.add_argument('-use_threshold',  action='store_false',
                        help='Disable p-value thresholding in Stage 2 (threshold applied by default)')
    parser.add_argument('-threshold',      type=float, default=0.005,
                        help='P-value threshold for Stage 2 (default: 5e-8)')
    parser.add_argument('-chunksize',      type=int, default=500_000,
                        help='Chunk size for reading GWAS files (default: 500000)')

    args = parser.parse_args()

    # ---- Parse arguments ----
    output_formats = [f.strip() for f in args.output_formats.split(',')]
    valid_formats  = {'npy', 'txt', 'gen'}
    bad_formats    = set(output_formats) - valid_formats
    if bad_formats:
        print(f"Error: Invalid output formats: {bad_formats}. Valid: {valid_formats}")
        return 1

    diseases = (list(DISEASE_CONFIGS.keys())
                if args.diseases.lower() == 'all'
                else [d.strip() for d in args.diseases.split(',')])

    unknown_diseases = [d for d in diseases if d not in DISEASE_CONFIGS]
    if unknown_diseases:
        print(f"Error: Unknown disease keys: {unknown_diseases}")
        print(f"Valid keys: {list(DISEASE_CONFIGS.keys())}")
        return 1

    if '-' in args.chromosomes:
        start, end = map(int, args.chromosomes.split('-'))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]

    # Output directories
    snv_output_dir  = os.path.join(args.output_base, "snv_filtered")
    gwas_output_dir = os.path.join(args.output_base, "gwas_filtered")
    Path(snv_output_dir).mkdir(parents=True, exist_ok=True)
    Path(gwas_output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Print configuration ----
    print("\n" + "=" * 80)
    print("UKBB TWO-STAGE SNP FILTERING PIPELINE")
    print("=" * 80)
    print(f"\nStage 1 — SNV Filtering")
    print(f"  Criteria: len(ref)==1, len(alt)==1, no commas, no missing")
    print(f"\nStage 2 — GWAS Summary Stats Filtering")
    print(f"  Matching: chromosome + bp position + alleles (Tier 2), position-only fallback (Tier 3)")
    print(f"  P-value threshold: {args.threshold if args.use_threshold else 'disabled'}")
    print(f"\nConfiguration:")
    print(f"  UKBB base path:   {args.base_path}")
    print(f"  Output base:      {args.output_base}")
    print(f"  Diseases:         {', '.join(DISEASE_CONFIGS[d]['name'] for d in diseases)}")
    print(f"  Chromosomes:      {chromosomes}")
    print(f"  Output formats:   {', '.join(output_formats)}")
    print(f"  GWAS chunk size:  {args.chunksize:,}")

    total_start    = time.time()
    snv_stats_list = []
    all_gwas_stats = []

    # =========================================================================
    # MAIN LOOP — one chromosome at a time
    # =========================================================================
    for chr_num in chromosomes:
        print(f"\n{'=' * 80}")
        print(f"CHROMOSOME {chr_num}")
        print(f"{'=' * 80}")

        # ---- Stage 1: SNV filtering ----
        try:
            snv_variants, snv_indices, variant_types = stage1_snv_filter(
                chr_num, args.base_path, snv_output_dir, output_formats
            )
            snv_stats_list.append({
                'chr':                  chr_num,
                'total_snps':           variant_types['total'],
                'snv_kept':             variant_types['snv_kept'],
                'removed':              variant_types['removed'],
                'retention_pct':        variant_types['retention_pct'],
                'indels_removed':       variant_types['indels_removed'],
                'multiallelic_removed': variant_types['multiallelic_removed'],
                'invalid_removed':      variant_types['invalid_removed'],
                'success':              True,
            })
        except Exception as exc:
            import traceback
            print(f"  ERROR in Stage 1 for chr{chr_num}: {exc}")
            traceback.print_exc()
            snv_stats_list.append({'chr': chr_num, 'success': False, 'error': str(exc)})
            # Cannot run Stage 2 without Stage 1 result
            for disease_key in diseases:
                all_gwas_stats.append({
                    'chr': chr_num, 'disease': disease_key,
                    'success': False, 'error': f'Stage 1 failed: {exc}'
                })
            gc.collect()
            continue

        # ---- Stage 2: GWAS filtering (per disease) ----
        for disease_key in diseases:
            try:
                gwas_stats = stage2_gwas_filter(
                    chr_num, disease_key,
                    snv_variants, snv_indices,
                    gwas_output_dir,
                    args.use_threshold, args.threshold,
                    output_formats, args.chunksize
                )
                all_gwas_stats.append(gwas_stats)
            except Exception as exc:
                import traceback
                print(f"  ERROR in Stage 2 ({disease_key}) for chr{chr_num}: {exc}")
                traceback.print_exc()
                all_gwas_stats.append({
                    'chr': chr_num, 'disease': disease_key,
                    'success': False, 'error': str(exc)
                })

        # Free memory before next chromosome
        del snv_variants, snv_indices
        gc.collect()

    # =========================================================================
    # SUMMARY FILES
    # =========================================================================
    create_snv_summary(snv_stats_list, snv_output_dir)
    create_gwas_summary(all_gwas_stats, diseases, gwas_output_dir,
                        args.threshold, args.use_threshold)

    total_elapsed = time.time() - total_start

    # =========================================================================
    # FINAL CONSOLE SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Stage 1
    snv_ok = [s for s in snv_stats_list if s.get('success')]
    if snv_ok:
        total_orig = sum(s['total_snps'] for s in snv_ok)
        total_snv  = sum(s['snv_kept']   for s in snv_ok)
        overall_ret = 100.0 * total_snv / total_orig if total_orig > 0 else 0.0
        print(f"\nStage 1 — SNV Filter ({len(snv_ok)} chromosomes):")
        print(f"  {'Chr':<5} {'Total':<14} {'SNVs Kept':<14} {'%':<8} "
              f"{'Indels':<12} {'Multi':<10}")
        print(f"  {'-'*65}")
        for s in sorted(snv_ok, key=lambda x: x['chr']):
            print(f"  {s['chr']:<5} {s['total_snps']:<14,} {s['snv_kept']:<14,} "
                  f"{s['retention_pct']:<8.2f} {s['indels_removed']:<12,} "
                  f"{s['multiallelic_removed']:<10,}")
        print(f"  {'-'*65}")
        print(f"  {'Total':<5} {total_orig:<14,} {total_snv:<14,} {overall_ret:<8.2f}")

    # Stage 2
    gwas_ok = [s for s in all_gwas_stats if s.get('success')]
    if gwas_ok:
        print(f"\nStage 2 — GWAS Filter ({len(gwas_ok)} chromosome-disease combinations):")
        for disease_key in diseases:
            d_stats = [s for s in gwas_ok if s.get('disease') == disease_key]
            if not d_stats:
                continue
            disease_name = DISEASE_CONFIGS[disease_key]['name']
            t_snv     = sum(s['original_ukbb_snps'] for s in d_stats)
            t_matched = sum(s['matched_snps'] for s in d_stats)
            t_ret     = 100.0 * t_matched / t_snv if t_snv > 0 else 0.0
            print(f"\n  {disease_name}:")
            print(f"    {'Chr':<5} {'SNV SNPs':<12} {'GWAS Before':<13} "
                  f"{'GWAS After':<12} {'Matched':<10} {'%':<8}")
            print(f"    {'-'*65}")
            for s in sorted(d_stats, key=lambda x: x.get('chr', 999)):
                print(f"    {s['chr']:<5} "
                      f"{s['original_ukbb_snps']:<12,} "
                      f"{s['original_gwas_snps_before_threshold']:<13,} "
                      f"{s['original_gwas_snps_after_threshold']:<12,} "
                      f"{s['matched_snps']:<10,} "
                      f"{s['retention_rate']:<8.2f}")
            print(f"    {'-'*65}")
            print(f"    {'Total':<5} {t_snv:<12,} {'':13} {'':12} {t_matched:<10,} {t_ret:<8.2f}")

    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"\n✓ Complete!")
    print(f"  SNV outputs:  {snv_output_dir}")
    print(f"  GWAS outputs: {gwas_output_dir}\n")

    failed = [s for s in snv_stats_list + all_gwas_stats if not s.get('success')]
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())