# =============================================================================
# REPLICATION ANALYSIS — BINARY PHENOTYPES
# UKBB BGEN + Python only (no plink)
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from bgen_reader import open_bgen
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

# Paths
BGEN_DIR    = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra"
SAMPLE_FILE = f"{BGEN_DIR}/ukb_imp_chr6_maf0.05.sample"
PHENO_FILE  = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_cancers_t2d_ukb676869_13102025.tsv"
SNP_DIR     = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/filtered_by_disease_gwas_summ_stats_5e8_threshold_top_snps" 
OUTPUT_DIR  = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/replication_results"

# Covariates to include in regression
COVARIATES = ['age', 'sex', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

# Binary phenotypes only
#BINARY_PHENOS = ['T2D']
BINARY_PHENOS = ['PrC', 'PanC', 'CRC', 'BC', 'T2D']

# Per-phenotype column mappings
# After running inspect_snp_files.py, fill in the actual column names here
# For each phenotype:
#   rsid_col  : column containing the rs ID
#   chr_col   : column containing chromosome number
#   ea_col    : effect allele (the allele the beta refers to)
#   oa_col    : other/non-effect allele (set to None if not present)
#   beta_col  : effect size (log-OR for binary traits)
#   p_col     : p-value from published GWAS

COLUMN_MAPS = {
    'CRC': {
        'rsid_col': 'rsid',     
        'chr_col':  'chromosome',
        'ea_col':   'Allele1',
        'oa_col':   'Allele2',      
        'beta_col': 'Effect',
        'p_col':    'P.value'
    },
    'BC': {
        'rsid_col': 'rsid',
        'chr_col':  'chromosome',
        'ea_col':   'a1',
        'oa_col':   'a0',
        'beta_col': 'bcac_onco_icogs_gwas_beta',
        'p_col':    'bcac_onco_icogs_gwas_P1df'
    },
    'PrC': {
        'rsid_col': 'rsid',
        'chr_col':  'chromosome',
        'ea_col':   'Allele1',
        'oa_col':   'Allele2',
        'beta_col': 'Effect',
        'p_col':    'Pvalue'
    },
    'PanC': {
        'rsid_col': 'rsid',
        'chr_col':  'chromosome',
        'ea_col':   'ALLELE1',
        'oa_col':   'ALLELE0',
        'beta_col': 'BETA',
        'p_col':    'P_LINREG'
    },
    'T2D': {
        'rsid_col': 'rsid',
        'chr_col':  'chromosome',
        'ea_col':   'EA',
        'oa_col':   'NEA',
        'beta_col': 'Beta',
        'p_col':    'Pvalue'
    },
}

# =============================================================================
# STEP 1: LOAD AND VALIDATE PHENOTYPE + SAMPLE DATA
# =============================================================================

print("="*60)
print("STEP 1: Loading phenotype and sample files")
print("="*60)

# Load phenotype file
pheno_df = pd.read_csv(PHENO_FILE, sep="\t")

# Drop the type-descriptor row (the "0 0 0 D ..." row) - check as integer first
pheno_df = pheno_df[pheno_df['ID_1'] != 0].copy()
pheno_df['ID_1'] = pheno_df['ID_1'].astype(str)

# Convert sex to numeric (1/2 → 0/1 or keep as is)
if 'sex' in pheno_df.columns and pheno_df['sex'].dtype == 'object':
    pheno_df['sex'] = pd.to_numeric(pheno_df['sex'], errors='coerce')

print(f"Phenotype file loaded: {pheno_df.shape[0]} samples")

# Quick check on case/control counts
print("\nCase/control counts:")
for pheno in BINARY_PHENOS:
    if pheno in pheno_df.columns:
        counts = pheno_df[pheno].value_counts(dropna=False)
        print(f"  {pheno}: {counts.to_dict()}")
    else:
        print(f"  {pheno}: COLUMN NOT FOUND IN PHENOTYPE FILE")

# Load .sample file — this defines the order of samples in all BGEN files
print(f"\nLoading sample file: {SAMPLE_FILE}")
sample_df = pd.read_csv(SAMPLE_FILE, sep=r"\s+", header=0, skiprows=[1])  # Skip type descriptor row
sample_df = sample_df.rename(columns={sample_df.columns[0]: 'ID_1',
                                       sample_df.columns[1]: 'ID_2'})
sample_df['ID_1'] = sample_df['ID_1'].astype(str)
print(f"Sample file loaded: {len(sample_df)} samples")

# Merge phenotype onto sample order — MUST preserve BGEN row order
# Left merge keeps all BGEN samples in order; unmatched get NaN phenotype
merged = sample_df[['ID_1']].merge(pheno_df, on='ID_1', how='left')

n_matched = merged['ID_1'].isin(pheno_df['ID_1']).sum()
print(f"Samples matched between BGEN and phenotype file: {n_matched}/{len(sample_df)}")

if n_matched == 0:
    raise ValueError("No samples matched! Check that ID_1 in phenotype file "
                     "matches ID format in .sample file")

# =============================================================================
# STEP 2: LOAD SNP FILES AND VALIDATE
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Loading SNP files")
print("="*60)

snp_data = {}

for pheno in BINARY_PHENOS:
    snp_file = f"{SNP_DIR}/{pheno}/{pheno}_filtered_snps_asc.csv"
    cmap = COLUMN_MAPS[pheno]
    
    try:
        df = pd.read_csv(snp_file)
        
        # Validate all required columns exist
        required = [cmap['rsid_col'], cmap['chr_col'],
                    cmap['ea_col'], cmap['beta_col'], cmap['p_col']]
        missing_cols = [c for c in required if c not in df.columns]
        
        if missing_cols:
            print(f"  {pheno}: MISSING COLUMNS {missing_cols} — "
                  f"available: {df.columns.tolist()}")
            continue
        
        # Standardise chromosome column — remove 'chr' prefix if present
        df[cmap['chr_col']] = df[cmap['chr_col']].astype(str).str.replace('chr','',
                               case=False).astype(int)
        
        snp_data[pheno] = df
        print(f"  {pheno}: {len(df)} SNPs loaded, "
              f"chromosomes: {sorted(df[cmap['chr_col']].unique())}")
        
    except FileNotFoundError:
        print(f"  {pheno}: FILE NOT FOUND — {snp_file}")

# =============================================================================
# STEP 3: REGRESSION FUNCTION
# =============================================================================

def get_dosage_for_snp(bgen, rsid):
    """
    Extract dosage array for a given rsID from an open bgen object.
    Returns (dosage_array, effect_allele_in_bgen, other_allele_in_bgen) or None.
    Handles both soft-called probabilities and hard-called genotypes.
    """
    idx = np.where(bgen.rsids == rsid)[0]
    
    if len(idx) == 0:
        return None, None, None
    
    idx = idx[0]
    
    try:
        # Read genotype data
        probs = bgen.read(idx)  # Single variant index (cleaner than tuple)
        
        # Handle different data formats
        if probs.ndim == 1:
            # 1D array - likely hard genotypes
            dosage = probs.astype(float)
        elif probs.ndim == 2:
            if probs.shape[1] == 3:
                # Soft probabilities: shape (n_samples, 3) → P(AA), P(AB), P(BB)
                dosage = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
            elif probs.shape[1] == 1:
                # Single column - reshape and treat as hard genotypes
                dosage = probs[:, 0].astype(float)
            else:
                return None, None, None
        elif probs.ndim == 3:
            # Batched read: shape (n_samples, n_variants, 3) - squeeze if n_variants=1
            if probs.shape[1] == 1:
                probs = probs.squeeze(axis=1)
                dosage = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
            else:
                return None, None, None
        else:
            return None, None, None
        
        # Get alleles — UKBB format is "A,G"
        alleles = bgen.allele_ids[idx].split(",")
        allele_ref = alleles[0]   # reference allele (dosage=0)
        allele_alt = alleles[1]   # alternative allele (dosage=2)
        
        return dosage, allele_ref, allele_alt
    
    except Exception as e:
        return None, None, None


def run_logistic_regression(rsid, chrom, pheno, effect_allele,
                             pub_beta, pub_p, merged_df):
    """
    Extract SNP dosage from BGEN, align alleles, run logistic regression.
    Returns a result dictionary.
    """
    bgen_path = f"{BGEN_DIR}/ukb_imp_chr{chrom}_maf0.05.bgen"
    
    if not os.path.exists(bgen_path):
        return {'rsID': rsid, 'phenotype': pheno,
                'error': f"BGEN file not found: chr{chrom}"}
    
    try:
        with open_bgen(bgen_path, verbose=False) as bgen:
            dosage, allele_ref, allele_alt = get_dosage_for_snp(bgen, rsid)
        
        if dosage is None:
            return {'rsID': rsid, 'phenotype': pheno,
                    'error': 'SNP not found in BGEN or unable to extract dosage'}
        
        # Build regression dataframe
        # Ensure we have all required columns
        cols_needed = ['ID_1', pheno] + COVARIATES
        cols_available = [c for c in cols_needed if c in merged_df.columns]
        missing_cols = [c for c in cols_needed if c not in merged_df.columns]
        
        if missing_cols:
            return {'rsID': rsid, 'phenotype': pheno,
                    'error': f'Missing columns: {missing_cols}. Available: {merged_df.columns.tolist()}'}
        
        reg_df = merged_df[cols_available].copy()
        
        # Handle case where BGEN file has fewer samples than expected
        if len(dosage) < len(reg_df):
            reg_df = reg_df.iloc[:len(dosage)].copy()
        elif len(dosage) > len(reg_df):
            return {'rsID': rsid, 'phenotype': pheno,
                    'error': f'Dosage length ({len(dosage)}) > data rows ({len(reg_df)})'}
        
        reg_df['dosage'] = dosage
        reg_df = reg_df.dropna(subset=[pheno, 'dosage'] + COVARIATES)
        reg_df[pheno]    = reg_df[pheno].astype(float)
        reg_df['dosage'] = reg_df['dosage'].astype(float)
        
        n_cases    = int((reg_df[pheno] == 1).sum())
        n_controls = int((reg_df[pheno] == 0).sum())
        
        if n_cases < 10 or n_controls < 10:
            return {'rsID': rsid, 'phenotype': pheno,
                    'error': f'Too few cases ({n_cases}) or controls ({n_controls})'}
        
        # Check allele frequency — warn if very different from expected
        maf_ukbb = reg_df['dosage'].mean() / 2
        
        # Determine if dosage needs flipping to match published effect allele
        # UKBB bgen: dosage counts allele_alt copies
        # If published effect allele matches allele_ref, flip dosage
        ea_upper  = effect_allele.upper()
        alt_upper = allele_alt.upper()
        ref_upper = allele_ref.upper()
        
        allele_flipped = False
        if ea_upper == ref_upper:
            # Published effect allele is the REF in UKBB — flip dosage
            reg_df['dosage'] = 2.0 - reg_df['dosage']
            allele_flipped = True
        elif ea_upper == alt_upper:
            # Correct orientation — no flip needed
            allele_flipped = False
        else:
            # Allele mismatch — could be strand flip, try complement
            complement = {'A':'T','T':'A','C':'G','G':'C'}
            ea_comp = complement.get(ea_upper, '')
            if ea_comp == alt_upper:
                allele_flipped = False  # strand flip but same allele
            elif ea_comp == ref_upper:
                reg_df['dosage'] = 2.0 - reg_df['dosage']
                allele_flipped = True
            else:
                return {'rsID': rsid, 'phenotype': pheno,
                        'error': f'Allele mismatch: published EA={ea_upper}, '
                                 f'BGEN alleles={ref_upper}/{alt_upper}'}
        
        # Run logistic regression
        covar_str = " + ".join(COVARIATES)
        formula   = f"Q('{pheno}') ~ dosage + {covar_str}"
        
        try:
            model = smf.logit(formula, data=reg_df).fit(
                disp=False, maxiter=200, method='bfgs'
            )
        except Exception as e:
            return {'rsID': rsid, 'phenotype': pheno,
                    'error': f'Regression failed: {str(e)[:100]}. '
                             f'Available columns: {reg_df.columns.tolist()}'}
        
        beta_ukbb = model.params['dosage']
        se_ukbb   = model.bse['dosage']
        p_ukbb    = model.pvalues['dosage']
        or_ukbb   = np.exp(beta_ukbb)
        
        # Direction concordance with published
        direction_match = np.sign(beta_ukbb) == np.sign(pub_beta)
        
        return {
            'rsID':             rsid,
            'phenotype':        pheno,
            'CHR':              chrom,
            'BGEN_REF':         allele_ref,
            'BGEN_ALT':         allele_alt,
            'Published_EA':     effect_allele,
            'Allele_Flipped':   allele_flipped,
            'MAF_UKBB':         round(maf_ukbb, 4),
            'BETA_UKBB':        round(beta_ukbb, 5),
            'SE_UKBB':          round(se_ukbb, 5),
            'OR_UKBB':          round(or_ukbb, 4),
            'P_UKBB':           p_ukbb,
            'N_cases':          n_cases,
            'N_controls':       n_controls,
            'Published_BETA':   pub_beta,
            'Published_P':      pub_p,
            'Direction_Match':  direction_match,
            'Nominal_Sig':      p_ukbb < 0.05,
            'error':            None
        }
    
    except Exception as e:
        return {'rsID': rsid, 'phenotype': pheno, 'error': str(e)}


# =============================================================================
# STEP 4: RUN REGRESSIONS FOR ALL PHENOTYPES
# =============================================================================

print("\n" + "="*60)
print("STEP 4: Running regressions")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

for pheno in BINARY_PHENOS:
    
    if pheno not in snp_data:
        print(f"\n{pheno}: skipped (no SNP data loaded)")
        continue
    
    df   = snp_data[pheno]
    cmap = COLUMN_MAPS[pheno]
    
    print(f"\n{'-'*50}")
    print(f"Phenotype: {pheno}  ({len(df)} SNPs)")
    print(f"{'-'*50}")
    
    pheno_results = []
    
    for i, row in df.iterrows():
        rsid   = str(row[cmap['rsid_col']])
        chrom  = int(row[cmap['chr_col']])
        ea     = str(row[cmap['ea_col']])
        pb     = float(row[cmap['beta_col']])
        pp     = float(row[cmap['p_col']])
        
        result = run_logistic_regression(
            rsid=rsid, chrom=chrom, pheno=pheno,
            effect_allele=ea, pub_beta=pb, pub_p=pp,
            merged_df=merged
        )
        
        pheno_results.append(result)
        all_results.append(result)
        
        # Progress print
        if result['error'] is None:
            print(f"  {i+1:5d}/{len(df)} | {rsid} chr{chrom} | "
                  f"BETA={result['BETA_UKBB']:+.4f}  "
                  f"P={result['P_UKBB']:.3e}  "
                  f"Direction={result['Direction_Match']}")
        else:
            if i % 100 == 0:  # Print every 100th error to see progress
                print(f"  {i+1:5d}/{len(df)} | {rsid} chr{chrom} | ERROR: {result['error'][:50]}")
    
    # Save per-phenotype results
    pheno_df_res = pd.DataFrame(pheno_results)
    pheno_df_res.to_csv(f"{OUTPUT_DIR}/{pheno}_replication.csv", index=False)
    
    # Summary
    ok = pheno_df_res[pheno_df_res['error'].isna()]
    not_found = pheno_df_res[pheno_df_res['error'].str.contains('not found', case=False, na=False)]
    if len(ok) > 0:
        print(f"\n  Summary for {pheno}:")
        print(f"    SNPs tested:          {len(ok)}/{len(pheno_df_res)}")
        print(f"    SNPs not in BGEN:     {len(not_found)}")
        print(f"    Direction concordance: {ok['Direction_Match'].mean():.1%}")
        print(f"    Nominally sig (p<0.05): {ok['Nominal_Sig'].sum()}/{len(ok)}")
    else:
        print(f"\n  Summary for {pheno}:")
        print(f"    No SNPs tested successfully")
        print(f"    SNPs not in BGEN: {len(not_found)}/{len(pheno_df_res)}")


# =============================================================================
# STEP 5: SAVE COMBINED RESULTS
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Saving combined results")
print("="*60)

all_df = pd.DataFrame(all_results)
all_df.to_csv(f"{OUTPUT_DIR}/all_phenotypes_replication.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/all_phenotypes_replication.csv")


# =============================================================================
# STEP 6: PLOTS
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Generating plots")
print("="*60)

def plot_replication(pheno, results_df, output_dir):
    df = results_df[(results_df['phenotype'] == pheno) &
                    (results_df['error'].isna())].copy()
    
    if len(df) == 0:
        print(f"  {pheno}: no valid results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"{pheno} — Replication in UKBB", fontsize=14, fontweight='bold')
    
    colors = ['#2ecc71' if m else '#e74c3c' for m in df['Direction_Match']]
    
    # ── Plot 1: Effect size scatter ──────────────────────────────────────────
    ax = axes[0]
    ax.scatter(df['Published_BETA'], df['BETA_UKBB'],
               c=colors, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    lim = max(abs(df[['Published_BETA','BETA_UKBB']].values.flatten())) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, label='y = x')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel("Published BETA", fontsize=11)
    ax.set_ylabel("UKBB BETA", fontsize=11)
    ax.set_title("Effect Size Comparison")
    ax.legend(fontsize=9)
    
    # Add SNP labels for outliers
    for _, row in df.iterrows():
        if abs(row['BETA_UKBB'] - row['Published_BETA']) > 0.3:
            ax.annotate(row['rsID'], (row['Published_BETA'], row['BETA_UKBB']),
                        fontsize=7, alpha=0.7)
    
    # ── Plot 2: -log10(P) scatter ────────────────────────────────────────────
    ax2 = axes[1]
    pub_logp  = -np.log10(df['Published_P'].astype(float).clip(lower=1e-300))
    ukbb_logp = -np.log10(df['P_UKBB'].astype(float).clip(lower=1e-300))
    
    ax2.scatter(pub_logp, ukbb_logp, c=colors, s=60, alpha=0.8,
                edgecolors='white', linewidth=0.5)
    ax2.axhline(-np.log10(0.05), color='orange', linestyle='--',
                alpha=0.7, label='p=0.05')
    ax2.set_xlabel("-log10(P) Published", fontsize=11)
    ax2.set_ylabel("-log10(P) UKBB", fontsize=11)
    ax2.set_title("P-value Comparison")
    ax2.legend(fontsize=9)
    
    # # ── Plot 3: OR forest plot ───────────────────────────────────────────────
    # ax3 = axes[2]
    # y_pos   = np.arange(len(df))
    # or_ukbb = df['OR_UKBB'].values
    
    # # 95% CI from SE
    # ci_lower = np.exp(df['BETA_UKBB'].values - 1.96 * df['SE_UKBB'].values)
    # ci_upper = np.exp(df['BETA_UKBB'].values + 1.96 * df['SE_UKBB'].values)
    
    # ax3.errorbar(or_ukbb, y_pos,
    #              xerr=[or_ukbb - ci_lower, ci_upper - or_ukbb],
    #              fmt='o', color='steelblue', ecolor='lightblue',
    #              elinewidth=1.5, capsize=3, markersize=5)
    # ax3.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='OR=1 (null)')
    # ax3.set_yticks(y_pos)
    # ax3.set_yticklabels(df['rsID'].values, fontsize=7)
    # ax3.set_xlabel("Odds Ratio (UKBB)", fontsize=11)
    # ax3.set_title("Forest Plot — UKBB ORs")
    # ax3.legend(fontsize=9)
    
    # Legend for direction
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Direction match'),
                       Patch(facecolor='#e74c3c', label='Direction mismatch')]
    axes[0].legend(handles=legend_elements + [plt.Line2D([0],[0],
                   linestyle='--', color='k', alpha=0.4, label='y=x')],
                   fontsize=8)
    
    plt.tight_layout()
    out_path = f"{output_dir}/{pheno}_replication_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


for pheno in BINARY_PHENOS:
    plot_replication(pheno, all_df, OUTPUT_DIR)


# =============================================================================
# STEP 7: FINAL SUMMARY TABLE
# =============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary_rows = []

for pheno in BINARY_PHENOS:
    df = all_df[(all_df['phenotype'] == pheno) & (all_df['error'].isna())]
    err = all_df[(all_df['phenotype'] == pheno) & (all_df['error'].notna())]
    
    if len(df) == 0:
        continue
    
    summary_rows.append({
        'Phenotype':              pheno,
        'SNPs_tested':            len(df),
        'SNPs_failed':            len(err),
        'Direction_concordance':  f"{df['Direction_Match'].mean():.1%}",
        'Nominally_sig_p005':     f"{df['Nominal_Sig'].sum()}/{len(df)}",
        'Median_UKBB_P':          f"{df['P_UKBB'].median():.3e}",
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{OUTPUT_DIR}/summary_table.csv", index=False)

print("\nDone. All outputs saved to:", OUTPUT_DIR)