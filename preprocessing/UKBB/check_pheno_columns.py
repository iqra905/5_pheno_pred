#!/usr/bin/env python3
"""Quick script to check phenotype file columns"""
import pandas as pd

PHENO_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_cancers_t2d_ukb676869_13102025.tsv"

print("=" * 80)
print("PHENOTYPE FILE COLUMNS")
print("=" * 80)

pheno_df = pd.read_csv(PHENO_FILE, sep="\t", nrows=5)
print(f"\nColumns in phenotype file ({len(pheno_df.columns)} total):")
print(pheno_df.columns.tolist())

print(f"\nData types:")
print(pheno_df.dtypes)

print(f"\nFirst few rows:")
print(pheno_df.head())

# Check for covariates
covariates_to_check = ['age', 'sex', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'Age', 'Sex', 'age_at_assessment', 'sex_f31_0_0']
print(f"\nChecking for common covariate names:")
for cov in covariates_to_check:
    if cov in pheno_df.columns:
        print(f"  ✓ Found: {cov}")

print("\n" + "=" * 80)
