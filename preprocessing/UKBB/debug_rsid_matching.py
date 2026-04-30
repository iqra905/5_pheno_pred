#!/usr/bin/env python3
"""
Debug script to check rsID matching between SNP file and BGEN file
"""
import pandas as pd
import numpy as np
from bgen_reader import open_bgen

# Load SNP file
SNP_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/filtered_by_disease_gwas_summ_stats_5e8_threshold_top_snps_pre/T2D/T2D_filtered_snps.csv"
BGEN_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_imp_chr1_maf0.05.bgen"

print("=" * 80)
print("Checking RSIDs in SNP file")
print("=" * 80)

snp_df = pd.read_csv(SNP_FILE)
print(f"SNP file shape: {snp_df.shape}")
print(f"Columns: {snp_df.columns.tolist()}")
print(f"\nFirst 10 rsIDs from SNP file:")
print(snp_df['rsid'].head(10).values)

# Filter for chr1 SNPs
snp_chr1 = snp_df[snp_df['chromosome'] == 1]['rsid'].values
print(f"\nTotal chr1 SNPs: {len(snp_chr1)}")
print(f"First 10 chr1 SNPs: {snp_chr1[:10]}")

print("\n" + "=" * 80)
print("Checking RSIDs in BGEN file")
print("=" * 80)

with open_bgen(BGEN_FILE, verbose=False) as bgen:
    print(f"BGEN has {bgen.nvariants} variants")
    print(f"\nFirst 10 rsIDs from BGEN file:")
    print(bgen.rsids[:10])
    
    print(f"\nAll rsIDs type: {type(bgen.rsids[0])}")
    
    # Check if any of our SNPs are in the BGEN
    bgen_rsids_set = set(bgen.rsids)
    snp_chr1_set = set(snp_chr1)
    
    matches = bgen_rsids_set.intersection(snp_chr1_set)
    print(f"\n✓ Matching rsIDs between SNP file and BGEN: {len(matches)}")
    if matches:
        print(f"  Examples: {list(matches)[:5]}")
    
    # Check if rsID is just empty or different format
    non_rs_count = sum(1 for r in bgen.rsids if not str(r).startswith('rs'))
    print(f"\nNon-'rs' prefixed variants in BGEN: {non_rs_count}")
    if non_rs_count > 0:
        print(f"  Examples: {[str(r) for r in bgen.rsids if not str(r).startswith('rs')][:5]}")

print("\n" + "=" * 80)
