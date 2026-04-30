# =============================================================================
# GWAS — PER-SNP LOGISTIC REGRESSION ON BGEN DATA WITH TRAIN/TEST SPLIT
# =============================================================================

import os
os.environ['OMP_NUM_THREADS']     = '1'
os.environ['MKL_NUM_THREADS']     = '1'
os.environ['OPENBLAS_NUM_THREADS']= '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS']    = '1'

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import glob
import time
import fcntl
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='GWAS logistic regression on BGEN data')
parser.add_argument('-bgen_dir', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra',
                    help='Directory containing chromosome-wise BGEN files')
parser.add_argument('-bgen_pattern', type=str, default='ukb_imp_chr{}_maf0.05.bgen',
                    help='BGEN filename pattern with {} for chromosome number')
parser.add_argument('-sample_file', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_imp_chr6_maf0.05.sample',
                    help='Path to .sample file defining sample order in BGEN files')
parser.add_argument('-phenotype_path', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_cancers_t2d_ukb676869_13102025.tsv',
                    help='Path to the phenotype TSV file')
parser.add_argument('-phenotype_column', type=str, default='T2D', help='Name of the binary phenotype column to analyze')
parser.add_argument('-output_dir', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/gwas_results/t2d', help='Directory for saving output files')
parser.add_argument('-covariates', type=str, default='age,gender,PC1,PC2,PC3,PC4,PC5,PC6', help='Comma-separated list of covariate column names')
parser.add_argument('-num_processes', type=int, default=10, help='Number of processes for parallel computation')
parser.add_argument('-p_value_threshold', type=float, default=0.05, help='P-value threshold for genome-wide significance')
parser.add_argument('-train_split', type=float, default=0.8, help='Fraction of samples for training (default: 0.8)')
parser.add_argument('-random_seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('-chromosomes', type=str, default='22', help='Chromosomes to analyze: "all", comma-separated (e.g. "1,3,5"), or range (e.g. "1-5")')
parser.add_argument('-batch_size', type=int, default=100, help='Number of SNPs to load per batch (adjust based on memory)')

args = parser.parse_args()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def open_bgen_with_lock(bgen_path, verbose=False):
    """
    Open a BGEN file with a file lock to prevent multiple jobs from
    creating the metadata cache simultaneously (race condition).
    bgen_reader is imported lazily here so that worker processes that
    import this module never trigger bgen_reader's background threads.
    """
    from bgen_reader import open_bgen
    lock_path = bgen_path + ".lock"
    with open(lock_path, 'w') as lock_file:
        print(f"Acquiring lock for {os.path.basename(bgen_path)}...")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            bgen = open_bgen(bgen_path, verbose=verbose)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    print(f"Lock released, BGEN opened successfully.")
    return bgen


def parse_chromosome_selection(selection):
    """Parse chromosome selection string into a list of chromosome identifiers."""
    if selection.lower() == 'all':
        return None
    if '-' in selection and ',' not in selection:
        start, end = selection.split('-')
        return [str(i) for i in range(int(start), int(end) + 1)]
    return [c.strip() for c in selection.split(',')]


def load_and_merge_samples(sample_file, phenotype_path, phenotype_column, covariates):
    """
    Load .sample file and phenotype TSV, merge them preserving BGEN row order.
    """
    sample_df = pd.read_csv(sample_file, sep=r"\s+", header=0, skiprows=[1])
    sample_df = sample_df.rename(columns={sample_df.columns[0]: 'ID_1',
                                           sample_df.columns[1]: 'ID_2'})
    sample_df['ID_1'] = sample_df['ID_1'].astype(str)
    print(f"Sample file loaded: {len(sample_df)} samples")

    pheno_df = pd.read_csv(phenotype_path, sep="\t")
    pheno_df = pheno_df[pheno_df['ID_1'] != 0].copy()
    pheno_df['ID_1'] = pheno_df['ID_1'].astype(str)
    print(f"Phenotype file loaded: {len(pheno_df)} samples")

    cols_needed = ['ID_1', phenotype_column] + covariates
    cols_available = [c for c in cols_needed if c in pheno_df.columns]
    missing = [c for c in cols_needed if c not in pheno_df.columns]
    if missing:
        raise ValueError(f"Missing columns in phenotype file: {missing}. "
                         f"Available: {pheno_df.columns.tolist()}")

    merged = sample_df[['ID_1']].merge(pheno_df[cols_available], on='ID_1', how='left')
    n_matched = merged[phenotype_column].notna().sum()
    print(f"Samples with phenotype data: {n_matched}/{len(merged)}")

    counts = merged[phenotype_column].value_counts(dropna=False)
    print(f"Case/control counts for {phenotype_column}: {counts.to_dict()}")

    return merged


def create_train_test_split(merged_df, phenotype_column, covariates,
                            train_fraction, random_seed, output_dir):
    """
    Create train/test split based on samples that have valid phenotype + covariates.
    """
    required_cols = [phenotype_column] + covariates
    valid_mask = merged_df[required_cols].notna().all(axis=1)
    valid_indices = np.where(valid_mask.values)[0]
    print(f"Samples with complete data (phenotype + covariates): {len(valid_indices)}/{len(merged_df)}")

    if train_fraction == 1.0:
        train_indices = valid_indices
        test_indices = np.array([], dtype=int)
    else:
        train_indices, test_indices = train_test_split(
            valid_indices,
            train_size=train_fraction,
            random_state=random_seed,
            shuffle=True
        )
        train_indices.sort()
        test_indices.sort()

    np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    print(f"Train/test split: {len(train_indices)} training, {len(test_indices)} test samples")

    return train_indices, test_indices


# Global variables shared with workers via fork (copy-on-write).
_cov_matrix = None
_phenotype  = None

def _worker_init(cov_matrix, phenotype):
    global _cov_matrix, _phenotype
    _cov_matrix = cov_matrix
    _phenotype  = phenotype


def perform_logistic_regression(task_args):
    """
    Perform logistic regression for a single SNP.
    task_args: (dosage, snp_index, rsid)
    Returns: (snp_index, rsid, p_value, beta, se)
    """
    dosage, snp_index, rsid = task_args
    try:
        X_snp = dosage.reshape(-1, 1)

        if _cov_matrix is not None and _cov_matrix.shape[1] > 0:
            X = np.concatenate((X_snp, _cov_matrix), axis=1)
        else:
            X = X_snp

        X = sm.add_constant(X)
        model = sm.Logit(_phenotype, X)
        result = model.fit(disp=0)

        snp_beta = result.params.iloc[1]
        snp_se   = result.bse.iloc[1]
        snp_p    = result.pvalues.iloc[1]

        return snp_index, rsid, snp_p, snp_beta, snp_se

    except Exception:
        return snp_index, rsid, 1.0, np.nan, np.nan


def process_chromosome(chrom, bgen_path, train_indices, phenotype_array,
                       covariate_matrix, p_value_threshold, output_dir,
                       num_processes, batch_size):
    """
    Process all SNPs in one chromosome BGEN file.
    SNPs are processed in batches; a fresh Pool is created per batch.
    """
    print(f"\n{'='*60}")
    print(f"Processing chromosome {chrom}: {bgen_path}")
    print(f"{'='*60}")
    start_time = time.time()

    # Set globals before any forking so workers inherit them via copy-on-write.
    global _cov_matrix, _phenotype
    _cov_matrix = covariate_matrix
    _phenotype  = phenotype_array

    bgen = open_bgen_with_lock(bgen_path, verbose=False)
    try:
        total_snps = bgen.nvariants
        rsids = bgen.rsids
        print(f"Chromosome {chrom}: {total_snps:,} SNPs, {bgen.nsamples:,} samples in file")

        all_results = []
        processed = 0

        for batch_start in range(0, total_snps, batch_size):
            batch_end = min(batch_start + batch_size, total_snps)
            current_batch = batch_end - batch_start

            print(f"  Batch {batch_start:,}-{batch_end-1:,} ({current_batch:,} SNPs) ...")

            batch_rsids = rsids[batch_start:batch_end]

            batch_indices = np.arange(batch_start, batch_end)
            geno = bgen.read(batch_indices)

            if hasattr(geno, 'probabilities'):
                probs = geno.probabilities
            else:
                probs = geno

            if probs.ndim == 2:
                probs = probs[:, np.newaxis, :]

            dosage_all   = probs[:, :, 1] + probs[:, :, 2] * 2.0
            dosage_train = dosage_all[train_indices, :]

            del probs, dosage_all

            process_args = [
                (dosage_train[:, i], batch_start + i, batch_rsids[i])
                for i in range(current_batch)
            ]

            del dosage_train

            with Pool(processes=num_processes) as pool:
                chunk_size = max(1, current_batch // (num_processes * 4))
                results = pool.map(perform_logistic_regression, process_args,
                                   chunksize=chunk_size)

            all_results.extend(results)
            processed += current_batch
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    Progress: {processed:,}/{total_snps:,} SNPs "
                  f"({elapsed:.0f}s, {rate:.1f} SNPs/s)")

            del process_args

    finally:
        bgen.close()

    # Build results dataframe
    results_df = pd.DataFrame(all_results,
                              columns=['SNP_Index', 'rsID', 'P_Value', 'Beta', 'SE'])
    results_df['Chromosome'] = chrom
    results_df['Is_Significant'] = (results_df['P_Value'] <= p_value_threshold) & (results_df['P_Value'] != 1.0)
    results_df['Failed'] = results_df['P_Value'] == 1.0
    results_df = results_df.sort_values('SNP_Index')

    all_file = os.path.join(output_dir, f'train_set_all_snps_chr{chrom}.csv')
    results_df.to_csv(all_file, index=False)

    sig_df = results_df[results_df['Is_Significant']]
    if not sig_df.empty:
        sig_file = os.path.join(output_dir, f'train_set_significant_snps_chr{chrom}.csv')
        sig_df.to_csv(sig_file, index=False)

    failed_count = results_df['Failed'].sum()
    if failed_count > 0:
        failed_df = results_df[results_df['Failed']]
        failed_file = os.path.join(output_dir, f'train_set_failed_snps_chr{chrom}.csv')
        failed_df.to_csv(failed_file, index=False)

    duration = time.time() - start_time
    print(f"\nChromosome {chrom} complete: {len(sig_df)} significant SNPs, "
          f"{failed_count} failed, {duration:.1f}s")

    return len(sig_df)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        print("="*60)
        print("GWAS — Per-SNP Logistic Regression (BGEN data)")
        print("="*60)
        overall_start = time.time()

        covariates = [c.strip() for c in args.covariates.split(',')]
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        print(f"Phenotype: {args.phenotype_column}")
        print(f"Covariates: {covariates}")

        # --- Step 1: Load and merge phenotype + sample data ---
        merged = load_and_merge_samples(
            args.sample_file, args.phenotype_path,
            args.phenotype_column, covariates
        )

        # --- Step 2: Train/test split ---
        train_indices, test_indices = create_train_test_split(
            merged, args.phenotype_column, covariates,
            args.train_split, args.random_seed, args.output_dir
        )

        # --- Step 3: Prepare phenotype and covariate arrays for training set ---
        phenotype_array = merged[args.phenotype_column].values[train_indices].astype(float)

        cov_data = merged[covariates].values[train_indices].astype(float)
        scaler = StandardScaler()
        covariate_matrix = scaler.fit_transform(cov_data)

        print(f"Phenotype array shape: {phenotype_array.shape}")
        print(f"Covariate matrix shape: {covariate_matrix.shape}")

        # --- Step 4: Determine which chromosomes to process ---
        selected_chrs = parse_chromosome_selection(args.chromosomes)

        if selected_chrs is None:
            pattern = os.path.join(args.bgen_dir, '*.bgen')
            bgen_files = sorted(glob.glob(pattern))
            chr_to_file = {}
            for f in bgen_files:
                base = os.path.basename(f)
                parts = base.replace('ukb_imp_chr', '').split('_')[0].split('.')[0]
                chr_to_file[parts] = f
        else:
            chr_to_file = {}
            for c in selected_chrs:
                bgen_name = args.bgen_pattern.format(c)
                bgen_path = os.path.join(args.bgen_dir, bgen_name)
                if os.path.exists(bgen_path):
                    chr_to_file[c] = bgen_path
                else:
                    print(f"WARNING: BGEN file not found for chromosome {c}: {bgen_path}")

        def chr_sort_key(c):
            try:
                return int(c)
            except ValueError:
                return {'X': 23, 'Y': 24, 'M': 25}.get(c, 999)

        sorted_chrs = sorted(chr_to_file.keys(), key=chr_sort_key)
        print(f"\nChromosomes to process: {sorted_chrs}")

        # --- Step 5: Process each chromosome ---
        total_significant = 0

        for chrom in sorted_chrs:
            n_sig = process_chromosome(
                chrom=chrom,
                bgen_path=chr_to_file[chrom],
                train_indices=train_indices,
                phenotype_array=phenotype_array,
                covariate_matrix=covariate_matrix,
                p_value_threshold=args.p_value_threshold,
                output_dir=args.output_dir,
                num_processes=args.num_processes,
                batch_size=args.batch_size
            )
            total_significant += n_sig

        # --- Summary ---
        total_time = time.time() - overall_start
        print(f"\n{'='*60}")
        print(f"GWAS COMPLETE")
        print(f"{'='*60}")
        print(f"Total significant SNPs (p < {args.p_value_threshold}): {total_significant}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Results saved in: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
