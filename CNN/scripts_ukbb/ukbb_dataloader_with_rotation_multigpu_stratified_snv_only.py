"""
Updated DataLoader for Chromosome-wise uint8 format GWAS data
WITH ROTATION-BASED CONTROL SUBSAMPLING FOR IMBALANCED MULTI-LABEL DATA
AND PREFILTERED SNV (SINGLE NUCLEOTIDE VARIANT) INDICES FOR FAST LOADING

New Features:
- RotatingControlSampler for handling severe class imbalance
- Rotates through control samples across epochs
- All controls used over multiple epochs
- Configurable control:case ratios
- FAST SNV-ONLY filtering using precomputed indices (no runtime computation!)
- Direct loading of SNV-only filtered indices from disk
- Use preprocess_snp_filtering.py to generate SNV-only filtered indices first
"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer


# Directory paths for chromosomes
BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8"
SNV_FILTERED_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/snv_filtered"


# Phenotype file path
PHENOTYPE_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/ukb_cancers_t2d_ukb676869_13102025_cleaned_matched.tsv"


# Global cache for SNV-only prefiltered indices (loaded once from disk)
_SNV_ONLY_PREFILTERED_INDICES_CACHE = {}

def verify_case_control_integrity(train_subjects, test_subjects, phenotype_data, disease_labels):
    """Check for overlap between splits and verify prevalence is preserved"""
    
    print("=" * 60)
    print("SPLIT INTEGRITY VERIFICATION")
    print("=" * 60)
    
    train_set = set(train_subjects)
    test_set  = set(test_subjects)
    overlap   = train_set & test_set
    
    print(f"  Train subjects: {len(train_set):,}")
    print(f"  Test subjects:  {len(test_set):,}")
    print(f"  Overlap:        {len(overlap)}  ← should be 0")
    
    def prevalence(subjects):
        rates = {}
        for label in disease_labels:
            vals = []
            for id1, id2 in subjects:
                row = phenotype_data[
                    (phenotype_data['ID_1'].astype(str) == str(id1)) &
                    (phenotype_data['ID_2'].astype(str) == str(id2))
                ]
                if len(row):
                    vals.append(row[label].values[0])
            rates[label] = np.mean(vals)
        return rates
    
    print("\n  Disease prevalence:")
    train_prev = prevalence(train_subjects)
    test_prev  = prevalence(test_subjects)
    print(f"  {'Label':<10} {'Train':>8} {'Test':>8}")
    for label in disease_labels:
        print(f"  {label:<10} {train_prev[label]:>8.4f} {test_prev[label]:>8.4f}")

def verify_genotypes(dataset, n_samples=5):
    """
    Check genotype tensors are the right shape, dtype,
    value range, and that SNV filtering reduced variant count.
    """
    print("=" * 60)
    print("GENOTYPE VERIFICATION")
    print("=" * 60)
    
    for i in range(min(n_samples, len(dataset))):
        id1, id2 = dataset.subject_list[i]
        genotypes, _, _ = dataset[i]
        g = genotypes.numpy()
        
        # Shape check: should be (n_snvs, 3)
        assert g.ndim == 2 and g.shape[1] == 3, \
            f"Unexpected genotype shape: {g.shape}"
        
        # Value range: uint8/255 → should be in [0, 1]
        assert g.min() >= 0.0 and g.max() <= 1.0, \
            f"Values out of [0,1] range: [{g.min()}, {g.max()}]"
        
        # No NaNs
        assert not np.isnan(g).any(), "NaNs found in genotype data!"
        
        print(f"  Subject {id1}_{id2}:")
        print(f"    Shape: {g.shape}  (n_snvs={g.shape[0]:,})")
        print(f"    Value range: [{g.min():.4f}, {g.max():.4f}]")
        print(f"    Non-zero entries: {(g > 0).sum():,} / {g.size:,} "
              f"({100*(g>0).mean():.1f}%)")
        
        # If SNV filtering is on, compare to unfiltered count
        if dataset.apply_snv_filter:
            raw = load_genotype_data_for_subject(id1, id2, apply_snv_filter=False)
            print(f"    Variants before SNV filter: {raw.shape[0]:,}")
            print(f"    Variants after  SNV filter: {g.shape[0]:,}  "
                  f"({100*g.shape[0]/raw.shape[0]:.1f}% retained)")

def verify_labels(dataset, phenotype_data, disease_labels, n_samples=20):
    """Cross-check returned labels against raw phenotype file values"""
    
    print("=" * 60)
    print("LABEL VERIFICATION")
    print("=" * 60)
    
    mismatches = 0
    for i in range(min(n_samples, len(dataset))):
        id1, id2 = dataset.subject_list[i]
        _, _, labels_tensor = dataset[i]
        returned_labels = labels_tensor.numpy()
        
        # Look up directly in phenotype dataframe
        pheno_row = phenotype_data[
            (phenotype_data['ID_1'].astype(str) == str(id1)) &
            (phenotype_data['ID_2'].astype(str) == str(id2))
        ]
        
        if len(pheno_row) == 0:
            print(f"  FAIL {id1}_{id2}: not found in phenotype data")
            continue
        
        expected = pheno_row[disease_labels].values[0].astype(float)
        
        if not np.allclose(returned_labels, expected, equal_nan=True):
            print(f"  MISMATCH {id1}_{id2}:")
            print(f"    Expected: {dict(zip(disease_labels, expected))}")
            print(f"    Got:      {dict(zip(disease_labels, returned_labels))}")
            mismatches += 1
        else:
            print(f"  OK {id1}_{id2}: {dict(zip(disease_labels, returned_labels))}")
    
    print(f"\nMismatches: {mismatches} / {min(n_samples, len(dataset))}")

def verify_covariates(dataset, n_samples=50):
    """
    Pull covariates for n_samples subjects and check distributions
    are sensible before and after normalisation.
    """
    print("=" * 60)
    print("COVARIATE VERIFICATION")
    print("=" * 60)
    
    ages, bmis, genders, pcs = [], [], [], []
    
    for i in range(min(n_samples, len(dataset))):
        _, covariates, _ = dataset[i]
        cov = covariates.numpy()
        
        # Assuming order: PC1-6, age, gender, BMI (matches covariate_dim logic)
        if dataset.use_covariates:
            pcs.append(cov[:6])
            offset = 6
        else:
            offset = 0
        if dataset.use_age:
            ages.append(cov[offset]); offset += 1
        if dataset.use_gender:
            genders.append(cov[offset]); offset += 1
        if dataset.use_bmi:
            bmis.append(cov[offset])
    
    def stats(name, vals):
        arr = np.array(vals)
        print(f"  {name}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}, "
              f"NaNs={np.isnan(arr).sum()}")
    
    if ages:   stats("Age (normalised)", ages)
    if bmis:   stats("BMI (normalised)", bmis)
    if genders: stats("Gender (normalised)", genders)
    if pcs:    stats("PC1 (normalised)", [p[0] for p in pcs])
    
    # Flag suspicious values
    all_cov = np.stack([dataset[i][1].numpy() for i in range(min(n_samples, len(dataset)))])
    nan_count = np.isnan(all_cov).sum()
    zero_rows = (all_cov == 0).all(axis=1).sum()
    print(f"\n  Total NaNs in covariates: {nan_count}")
    print(f"  All-zero covariate rows: {zero_rows} / {n_samples}")

def verify_id_matching(phenotype_data, n_samples=10):
    """Check that genotype filenames resolve to correct phenotype rows"""
    
    subjects = discover_subjects_from_phenotype(phenotype_data)
    sample = subjects[:n_samples]
    
    print("=" * 60)
    print("ID MATCHING VERIFICATION")
    print("=" * 60)
    
    for id1, id2 in sample:
        subject_key = f"{id1}_{id2}"
        
        # Check phenotype lookup
        pheno_match = phenotype_data[
            (phenotype_data['ID_1'].astype(str) == str(id1)) &
            (phenotype_data['ID_2'].astype(str) == str(id2))
        ]
        
        if len(pheno_match) == 0:
            print(f"  FAIL {subject_key}: No phenotype row found!")
        elif len(pheno_match) > 1:
            print(f"  WARN {subject_key}: Multiple phenotype rows ({len(pheno_match)}) — duplicates?")
        else:
            row = pheno_match.iloc[0]
            print(f"  OK   {subject_key} → age={row.get('age','?')}, "
                  f"gender={row.get('gender','?')}, BMI={row.get('BMI','?'):.1f}")
        
        # Check genotype file exists and has expected shape
        sample_files = get_sample_files_for_subject(id1, id2)
        if sample_files:
            chr1_data = np.load(sample_files[1])
            print(f"         Chr1 genotype shape: {chr1_data.shape}, "
                  f"dtype: {chr1_data.dtype}, "
                  f"value range: [{chr1_data.min()}, {chr1_data.max()}]")
        else:
            print(f"         FAIL: Missing genotype files!")


def load_snv_only_prefiltered_indices(chr_num, snv_filtered_path=SNV_FILTERED_PATH, rank=0):
    """
    Load SNV-ONLY prefiltered SNP indices from disk (computed by preprocess_snp_filtering.py)
    
    This function loads pre-computed SNV-only filter indices, eliminating the need for
    runtime computation. Much faster than computing indices on the fly!
    
    SNVs are single nucleotide variants (one reference allele, one alternate allele).
    Indels and other variants are excluded.
    
    Args:
        chr_num: Chromosome number (1-22)
        snv_filtered_path: Base directory path for SNV-only filtered indices
        rank: Process rank for distributed training (default 0)
    
    Returns:
        numpy array of indices to keep, or None if file doesn't exist
    """
    cache_key = f'chr{chr_num}_snv_only_prefiltered'
    
    # Check if already loaded in cache
    if cache_key in _SNV_ONLY_PREFILTERED_INDICES_CACHE:
        return _SNV_ONLY_PREFILTERED_INDICES_CACHE[cache_key]
    
    # Try to load SNV-only prefiltered indices
    chr_dir = os.path.join(snv_filtered_path, f"chr{chr_num}")
    
    # Try .npy format first (binary, faster)
    npy_file = os.path.join(chr_dir, f"chr{chr_num}_snv_filtered_indices.npy")
    txt_file = os.path.join(chr_dir, f"chr{chr_num}_snv_filtered_indices.txt")
    
    if os.path.exists(npy_file):
        if rank == 0:
            print(f"  Loading SNV-only prefiltered indices for chr{chr_num} (fast load!)")
        indices = np.load(npy_file)
        _SNV_ONLY_PREFILTERED_INDICES_CACHE[cache_key] = indices
        if rank == 0:
            print(f"    Loaded {len(indices):,} SNV-only filtered indices")
        return indices
    elif os.path.exists(txt_file):
        if rank == 0:
            print(f"  Loading SNV-only prefiltered indices for chr{chr_num} from text file...")
        indices = np.loadtxt(txt_file, dtype=np.int64)
        _SNV_ONLY_PREFILTERED_INDICES_CACHE[cache_key] = indices
        if rank == 0:
            print(f"    Loaded {len(indices):,} SNV-only filtered indices")
        return indices
    else:
        if rank == 0:
            print(f"     WARNING: No SNV-only prefiltered indices found for chr{chr_num}!")
            print(f"     Expected: {npy_file}")
            print(f"     Run preprocess_snp_filtering.py with SNV-only flag first to generate SNV-only filtered indices")
            print(f"     Will load ALL SNPs for this chromosome (no SNV filtering)")
        return None

def precompute_case_control_indices(train_subjects, phenotype_data, disease_labels, rank=0):
    """
    Efficiently identify case and control indices using vectorized operations
    
    Args:
        train_subjects: List of subject tuples
        phenotype_data: DataFrame with phenotype information
        disease_labels: List of disease label column names
        rank: Process rank for distributed training (default 0)
    
    Returns:
        case_indices: List of indices where subject has ≥1 disease
        control_indices: List of indices where subject has 0 diseases
    """
    if rank == 0:
        print("Pre-computing case/control indices (one-time operation)...")
    start_time = time.time()
    
    # Create a mapping dataframe for fast lookup
    train_df = pd.DataFrame(train_subjects, columns=['ID_1', 'ID_2'])
    train_df['idx'] = range(len(train_subjects))
    
    # Ensure ID types match
    train_df['ID_1'] = train_df['ID_1'].astype(str)
    train_df['ID_2'] = train_df['ID_2'].astype(str)
    phenotype_data_copy = phenotype_data.copy()
    phenotype_data_copy['ID_1'] = phenotype_data_copy['ID_1'].astype(str)
    phenotype_data_copy['ID_2'] = phenotype_data_copy['ID_2'].astype(str)
    
    # Merge with phenotype data
    merged = train_df.merge(
        phenotype_data_copy[['ID_1', 'ID_2'] + disease_labels],
        on=['ID_1', 'ID_2'],
        how='left'
    )
    
    # Compute sum of diseases for each subject
    merged['disease_sum'] = merged[disease_labels].sum(axis=1)
    
    # Identify cases and controls
    case_indices = merged[merged['disease_sum'] > 0]['idx'].tolist()
    control_indices = merged[merged['disease_sum'] == 0]['idx'].tolist()
    
    elapsed = time.time() - start_time
    if rank == 0:
        print(f"Pre-computation completed in {elapsed:.2f} seconds")
        print(f"  Cases (≥1 disease): {len(case_indices):,}")
        print(f"  Controls (0 diseases): {len(control_indices):,}")
        print(f"  Ratio: 1:{len(control_indices)/max(1, len(case_indices)):.1f}")
    
    return case_indices, control_indices

def get_sample_files_for_subject(id1, id2):
    """
    Get all 22 chromosome files for a given subject
    
    Args:
        id1: First ID from phenotype file (ID_1)
        id2: Second ID from phenotype file (ID_2)
    
    Returns:
        Dictionary mapping chromosome number to file path
        Returns None if any chromosome file is missing
    """
    sample_files = {}
    
    for chr_num in range(1, 23):  # Chromosomes 1-22
        chr_dir = os.path.join(BASE_PATH, f"chr{chr_num}")
        file_pattern = f"sample_{id1}_{id2}_chr{chr_num}.npy"
        file_path = os.path.join(chr_dir, file_pattern)
        
        if not os.path.exists(file_path):
            print(f"Warning: Missing file for subject {id1}_{id2}, chr{chr_num}: {file_path}")
            return None
        
        sample_files[chr_num] = file_path
    
    return sample_files

def discover_subjects_from_phenotype(phenotype_data, rank=0):
    """
    Discover subjects that are in phenotype file AND have complete chromosome data
    
    This is more efficient than discovering all subjects first, since it only checks
    subjects that will actually be used (those in the phenotype file).
    
    Args:
        phenotype_data: DataFrame with phenotype information
        rank: Process rank for distributed training (default 0)
    
    Returns:
        List of tuples: [(id1, id2), ...]
    """
    if rank == 0:
        print("Discovering subjects from phenotype file with complete chromosome data...")
    
    # Start with subjects in phenotype file
    phenotype_subjects = set()
    for _, row in phenotype_data.iterrows():
        # Convert to int first to remove .0 from float IDs (e.g., 1673761.0 -> 1673761)
        id1 = str(int(row['ID_1']))
        id2 = str(int(row['ID_2']))
        phenotype_subjects.add((id1, id2))
    
    if rank == 0:
        print(f"Found {len(phenotype_subjects):,} unique subjects in phenotype file")
    
    # Check which phenotype subjects have complete chromosome data
    subjects_with_complete_data = []
    missing_count = 0
    
    for id1, id2 in phenotype_subjects:
        sample_files = get_sample_files_for_subject(id1, id2)
        
        if sample_files is not None:
            subjects_with_complete_data.append((id1, id2))
        else:
            missing_count += 1
    
    if rank == 0:
        print(f"Found {len(subjects_with_complete_data):,} subjects with complete chromosome data")
        if missing_count > 0:
            print(f"  {missing_count:,} subjects in phenotype file missing chromosome data")
    
    return subjects_with_complete_data


def discover_all_subjects():
    """
    DEPRECATED: Use discover_subjects_from_phenotype() instead.
    This function is kept for backward compatibility.
    
    Discover all subjects that have complete data across all 22 chromosomes
    
    Returns:
        List of tuples: [(id1, id2), ...]
    """
    print("WARNING: discover_all_subjects() is deprecated. Use discover_subjects_from_phenotype() instead.")
    print("Discovering subjects with complete chromosome data...")
    
    # Start by scanning chr1 directory to get list of subjects
    chr1_dir = os.path.join(BASE_PATH, "chr1")
    chr1_files = glob.glob(os.path.join(chr1_dir, "sample_*_chr1.npy"))
    
    print(f"Found {len(chr1_files)} files in chr1 directory")
    
    subjects_with_complete_data = []
    
    for chr1_file in chr1_files:
        # Extract ID_1 and ID_2 from filename
        basename = os.path.basename(chr1_file)
        # Format: sample_ID1_ID2_chr1.npy
        parts = basename.replace("sample_", "").replace("_chr1.npy", "").split("_")
        
        if len(parts) == 2:
            id1, id2 = parts[0], parts[1]
            
            # Check if all 22 chromosomes exist for this subject
            sample_files = get_sample_files_for_subject(id1, id2)
            
            if sample_files is not None:
                subjects_with_complete_data.append((id1, id2))
    
    print(f"Found {len(subjects_with_complete_data):,} subjects with complete data (all 22 chromosomes)")
    return subjects_with_complete_data


def load_genotype_data_for_subject(id1, id2, return_shapes=False, apply_snv_filter=True, rank=0):
    """
    Load and concatenate genotype data from all 22 chromosomes for a subject
    WITH SNV-ONLY FILTERING using PREFILTERED INDICES (very fast!)
    
    Args:
        id1: First ID
        id2: Second ID
        return_shapes: If True, return chromosome shapes for debugging
        rank: Process rank for distributed training (default 0)
    
    Returns:
        Concatenated genotype array of shape (total_snv_variants, 3)
        If return_shapes=True, also returns list of (original_shape, filtered_shape) tuples per chromosome
    """
    sample_files = get_sample_files_for_subject(id1, id2)
    
    if sample_files is None:
        raise ValueError(f"Cannot load data for subject {id1}_{id2}: missing chromosome files")
    
    chromosome_data = []
    chromosome_shapes = []
    
    for chr_num in range(1, 23):
        file_path = sample_files[chr_num]
        
        # Load uint8 data
        chr_data = np.load(file_path)  # Shape: (n_snps_chr, 3)
        original_shape = chr_data.shape
        
        # Apply SNV-only filtering if requested (using prefiltered SNV-only indices)
        if apply_snv_filter:
            # Load SNV-only prefiltered indices (cached after first load)
            filter_indices = load_snv_only_prefiltered_indices(
                chr_num, 
                rank=rank)
            
            if filter_indices is not None:
                # Filter the data to keep only SNV variants
                chr_data = chr_data[filter_indices, :]
                filtered_shape = chr_data.shape
                
                if return_shapes:
                    chromosome_shapes.append((original_shape, filtered_shape))
            else:
                # No SNV-only prefiltered indices found - use all SNPs
                if return_shapes:
                    chromosome_shapes.append((original_shape, original_shape))
        else:
            if return_shapes:
                chromosome_shapes.append(original_shape)
        
        # Convert uint8 to float32
        chr_data = chr_data.astype(np.float32) / 255.0
        chromosome_data.append(chr_data)
    
    # Concatenate all chromosomes along SNP dimension
    genotype_data = np.concatenate(chromosome_data, axis=0)  # Shape: (total_snvs, 3)
    
    if return_shapes:
        return genotype_data, chromosome_shapes
    else:
        return genotype_data


def get_input_size_from_subject(id1, id2, apply_snv_filter=True):
    """
    Get the total number of SNVs for a subject by loading their data
    WITH SNV-ONLY FILTERING
    
    Args:
        id1: First ID
        id2: Second ID
        apply_snv_filter: If True, show SNV-only filtered sizes (default: True)
    
    Returns:
        Total number of SNVs across all chromosomes (after SNV-only filtering if applied)
    """
    genotype_data, shapes = load_genotype_data_for_subject(
        id1, id2, 
        return_shapes=True, 
        apply_snv_filter=apply_snv_filter
        )
    total_snvs = genotype_data.shape[0]
    
    print(f"\nSNV counts per chromosome for subject {id1}_{id2}:")
    
    if apply_snv_filter:
        total_original = 0
        total_filtered = 0
        for chr_num, (orig_shape, filt_shape) in enumerate(shapes, 1):
            total_original += orig_shape[0]
            total_filtered += filt_shape[0]
            pct_kept = 100 * filt_shape[0] / orig_shape[0] if orig_shape[0] > 0 else 0
            print(f"  Chr {chr_num}: {orig_shape[0]:,} → {filt_shape[0]:,} SNVs ({pct_kept:.1f}% SNVs kept)")
        
        print(f"\n  Total (Original): {total_original:,} variants")
        print(f"  Total (SNVs only): {total_filtered:,} SNVs")
        print(f"  Overall: {100*total_filtered/total_original:.1f}% SNVs retained")
    else:
        for chr_num, shape in enumerate(shapes, 1):
            print(f"  Chr {chr_num}: {shape[0]:,} SNVs")
        print(f"  Total: {total_snvs:,} SNVs")
    
    return total_snvs


class CovariateNormalizer:
    """Normalizer for covariates (Age, BMI, PCs, etc.)"""
    def __init__(self, method="standard"):
        self.method = method
        self.scaler = None
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif method == "power":
            self.scaler = PowerTransformer(method='yeo-johnson')   

    def fit(self, data):
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.scaler.fit(data)
    
    def transform(self, data):
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return self.scaler.transform(data)
        return data


# ROTATION-BASED CONTROL SUBSAMPLING

class RotatingControlSampler:
    """
    Rotates through control samples across epochs while keeping all cases.
    Designed for severe class imbalance in multi-label disease prediction.
    
    For multi-label setting:
    - Controls = subjects negative for ALL diseases
    - Cases = subjects positive for at least ONE disease
    """
    
    def __init__(self, case_subject_indices, control_subject_indices, 
                 target_ratio=5, n_rotations=None, random_state=42, stratify_by=None, rank=0):
        """
        Args:
            case_subject_indices: List of indices for case subjects
            control_subject_indices: List of indices for control subjects
            target_ratio: Desired controls:cases ratio (e.g., 5 means 5 controls per 1 case)
            n_rotations: Number of control subsets (auto-calculated if None)
            random_state: Random seed for reproducibility
            stratify_by: Optional dict mapping subject indices to strata (e.g., age groups)
            rank: Process rank for distributed training (default 0)
        """
        self.case_indices = np.array(case_subject_indices)
        self.control_indices = np.array(control_subject_indices)
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.stratify_by = stratify_by
        self.rank = rank  
        
        # Calculate controls needed per epoch
        self.n_cases = len(self.case_indices)
        self.controls_per_epoch = int(self.n_cases * target_ratio)
        
        # Calculate number of rotations needed to cover all controls
        self.n_controls = len(self.control_indices)
        
        if n_rotations is None:
            self.n_rotations = max(1, int(np.ceil(self.n_controls / self.controls_per_epoch)))
        else:
            self.n_rotations = n_rotations
        
        # Shuffle controls and split into rotation groups
        np.random.seed(random_state)
        shuffled_controls = np.random.permutation(self.control_indices)
        
        # Split into rotation groups
        self.control_groups = []
        for i in range(self.n_rotations):
            start_idx = i * self.controls_per_epoch
            end_idx = min((i + 1) * self.controls_per_epoch, self.n_controls)
            self.control_groups.append(shuffled_controls[start_idx:end_idx])
        
        # Convert to numpy arrays
        self.control_groups = [np.array(group) for group in self.control_groups]
        
        self.current_rotation = 0
        
        if rank == 0:
            print(f"\n{'='*80}")
            print("ROTATION-BASED CONTROL SAMPLER INITIALIZED")
            print(f"{'='*80}")
            print(f"  Total cases (≥1 disease): {self.n_cases:,}")
            print(f"  Total controls (0 diseases): {self.n_controls:,}")
            print(f"  Original imbalance ratio: 1:{self.n_controls/self.n_cases:.1f}")
            print(f"  Target ratio per epoch: 1:{target_ratio}")
            print(f"  Controls per epoch: {self.controls_per_epoch:,}")
            print(f"  Number of rotations: {self.n_rotations}")
            print(f"  Controls per rotation: {[len(g) for g in self.control_groups]}")
            print(f"  All {self.n_controls:,} controls will be used over {self.n_rotations} epochs")
            print(f"{'='*80}\n")
    
    def get_epoch_indices(self, epoch):
        """
        Get subject indices for a specific epoch
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Array of subject indices for this epoch (cases + rotated controls)
        """
        rotation_idx = epoch % self.n_rotations

        # All cases + current rotation's controls
        epoch_controls = self.control_groups[rotation_idx]
        epoch_indices = np.concatenate([self.case_indices, epoch_controls])
        
        # Shuffle within epoch for random batch composition
        np.random.shuffle(epoch_indices)
        
        return epoch_indices
    
    def get_rotation_info(self, epoch):
        """Get information about current rotation"""
        rotation_idx = epoch % self.n_rotations
        cycle_num = epoch // self.n_rotations
        epoch_in_cycle = (epoch % self.n_rotations) + 1
        
        return {
            'epoch': epoch,
            'rotation_idx': rotation_idx,
            'rotation_num': rotation_idx + 1,
            'total_rotations': self.n_rotations,
            'cycle_num': cycle_num,
            'epoch_in_cycle': epoch_in_cycle,
            'n_controls_in_rotation': len(self.control_groups[rotation_idx]),
            'n_cases': self.n_cases,
            'total_samples': self.n_cases + len(self.control_groups[rotation_idx]),
            'ratio': f"1:{len(self.control_groups[rotation_idx])/self.n_cases:.1f}"
        }
    
    def print_rotation_info(self, epoch):
        """Print formatted rotation information (only on rank 0)"""
        if self.rank == 0:
            info = self.get_rotation_info(epoch)
            print(f"\n{'─'*80}")
            print(f"EPOCH {info['epoch']} | Rotation {info['rotation_num']}/{info['total_rotations']} "
                  f"| Cycle {info['cycle_num']} | Epoch in Cycle: {info['epoch_in_cycle']}")
            print(f"  Samples: {info['n_cases']:,} cases + {info['n_controls_in_rotation']:,} controls "
                  f"= {info['total_samples']:,} total")
            print(f"  Ratio: {info['ratio']}")
            print(f"{'─'*80}")


class MultilabelGenotypeDataset(Dataset):
    """
    Dataset for chromosome-wise GWAS data with multi-label disease prediction
    
    - 22 .npy files per subject (one per chromosome)
    - uint8 format
    - Files: sample_ID1_ID2_chr*.npy
    - Split across two base directories
    - FILTERS TO SNVs ONLY (single nucleotide variants)
    """
    
    def __init__(self, subject_list, phenotype_data, disease_labels, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                 norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                 fit_normalizers=True, normalizers=None, apply_snv_filter=True, rank=0):
        """
        Args:
            subject_list: List of tuples [(id1, id2), ...] representing subjects
            phenotype_data: DataFrame with phenotype information
            disease_labels: List of disease column names to predict
            use_covariates: Whether to use PC covariates
            use_age: Whether to use age
            use_gender: Whether to use gender
            use_bmi: Whether to use BMI
            norm_age/norm_pcs/norm_gender/norm_bmi: Normalization method for each covariate
            fit_normalizers: Whether to fit normalizers (True for train, False for test)
            normalizers: Pre-fitted normalizers (for test set)
            apply_snv_filter: Whether to apply SNV-only filtering using prefiltered indices (default: True)
            rank: Process rank for distributed training (default: 0)
        """
        self.subject_list = subject_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.apply_snv_filter = apply_snv_filter
        self.rank = rank
        
        # Create subject ID to phenotype mapping
        self.phenotype_data['subject_key'] = (
            self.phenotype_data['ID_1'].astype(str) + '_' + 
            self.phenotype_data['ID_2'].astype(str)
        )
        self.pheno_dict = self.phenotype_data.set_index('subject_key').to_dict('index')
        
        # Initialize normalizers
        if normalizers is not None:
            # Use pre-fitted normalizers (test set)
            self.age_normalizer = normalizers.get('age')
            self.pc_normalizer = normalizers.get('pc')
            self.gender_normalizer = normalizers.get('gender')
            self.bmi_normalizer = normalizers.get('bmi')
        else:
            # Create new normalizers
            self.age_normalizer = CovariateNormalizer(norm_age) if use_age else None
            self.pc_normalizer = CovariateNormalizer(norm_pcs) if use_covariates else None
            self.gender_normalizer = CovariateNormalizer(norm_gender) if use_gender else None
            self.bmi_normalizer = CovariateNormalizer(norm_bmi) if use_bmi else None
            
            if fit_normalizers:
                self._fit_normalizers()
        
        # Calculate covariate dimension
        self.covariate_dim = 0
        if use_covariates:
            self.covariate_dim += 6  # 6 PCs
        if use_age:
            self.covariate_dim += 1
        if use_gender:
            self.covariate_dim += 1
        if use_bmi:
            self.covariate_dim += 1
        
        print(f"Dataset initialized with {len(self.subject_list)} subjects")
        print(f"  Disease labels: {disease_labels}")
        print(f"  Covariate dimension: {self.covariate_dim}")
        print(f"  Using: PCs={use_covariates}, Age={use_age}, Gender={use_gender}, BMI={use_bmi}")
        print(f"  SNV-only filtering: {apply_snv_filter}")
    
    def _fit_normalizers(self):
        """Fit normalizers on the training data"""
        print("Fitting normalizers on training data...")
        
        # Collect covariate data
        age_data = []
        pc_data = []
        gender_data = []
        bmi_data = []
        
        for id1, id2 in self.subject_list:
            subject_key = f"{id1}_{id2}"
            if subject_key in self.pheno_dict:
                pheno = self.pheno_dict[subject_key]
                
                if self.use_age and 'age' in pheno:
                    age_data.append(pheno['age'])
                
                if self.use_covariates:
                    pcs = [pheno.get(f'PC{i}', 0.0) for i in range(1, 7)]
                    pc_data.append(pcs)
                
                if self.use_gender and 'gender' in pheno:
                    gender_data.append(pheno['gender'])
                
                if self.use_bmi and 'BMI' in pheno:
                    bmi_data.append(pheno['BMI'])
        
        # Fit normalizers
        if self.age_normalizer and len(age_data) > 0:
            self.age_normalizer.fit(np.array(age_data))
        
        if self.pc_normalizer and len(pc_data) > 0:
            self.pc_normalizer.fit(np.array(pc_data))
        
        if self.gender_normalizer and len(gender_data) > 0:
            self.gender_normalizer.fit(np.array(gender_data))
        
        if self.bmi_normalizer and len(bmi_data) > 0:
            self.bmi_normalizer.fit(np.array(bmi_data))
        
        print("Normalizers fitted successfully")
    
    def get_normalizers(self):
        """Return fitted normalizers for use in test set"""
        return {
            'age': self.age_normalizer,
            'pc': self.pc_normalizer,
            'gender': self.gender_normalizer,
            'bmi': self.bmi_normalizer
        }
    
    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            genotype_tensor: Shape (n_snvs, 3)
            covariates_tensor: Shape (covariate_dim,)
            labels_tensor: Shape (n_diseases,)
        """
        id1, id2 = self.subject_list[idx]
        subject_key = f"{id1}_{id2}"
        
        # Get phenotype data
        if subject_key not in self.pheno_dict:
            raise ValueError(f"Subject {subject_key} not found in phenotype data")
        
        pheno = self.pheno_dict[subject_key]
        
        # Extract disease labels
        labels = [pheno.get(disease, 0) for disease in self.disease_labels]
        
        # Extract and normalize covariates
        covariates = []
        
        if self.use_covariates:
            pcs = np.array([pheno.get(f'PC{i}', 0.0) for i in range(1, 7)]).reshape(1, -1)
            if self.pc_normalizer:
                pcs = self.pc_normalizer.transform(pcs)
            covariates.extend(pcs.flatten())
        
        if self.use_age:
            age = np.array([pheno.get('age', 0.0)]).reshape(1, -1)
            if self.age_normalizer:
                age = self.age_normalizer.transform(age)
            covariates.extend(age.flatten())
        
        if self.use_gender:
            gender = np.array([pheno.get('gender', 0.0)]).reshape(1, -1)
            if self.gender_normalizer:
                gender = self.gender_normalizer.transform(gender)
            covariates.extend(gender.flatten())
        
        if self.use_bmi:
            bmi = np.array([pheno.get('BMI', 0.0)]).reshape(1, -1)
            if self.bmi_normalizer:
                bmi = self.bmi_normalizer.transform(bmi)
            covariates.extend(bmi.flatten())
        
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        
        # Load genotype data (with SNV-only filtering if enabled)
        genotype_data = load_genotype_data_for_subject(
            id1, id2, 
            return_shapes=False, 
            apply_snv_filter=self.apply_snv_filter, 
            rank=self.rank
            )
        genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor


def prepare_data_splits(disease_labels,
                        phenotype_file=PHENOTYPE_FILE,
                        test_size=0.2,
                        random_state=42,
                        rank=0,
                        use_rotation=False):
    """
    Prepare train/test splits for the dataset

    Stratification rules (when rotation is DISABLED):
    - Single-label: stratify by class
    - Multi-label: stratify by case/control (≥1 disease vs none)

    Args:
        disease_labels: List of disease column names
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        rank: Process rank for distributed training
        use_rotation: Whether rotation sampling will be used in training
    """
    if rank == 0:
        print("=" * 80)
        print("PREPARING DATA SPLITS")
        print("=" * 80)

    # Load phenotype data
    if rank == 0:
        print(f"\nLoading phenotype data from: {phenotype_file}")
    phenotype_data = pd.read_csv(phenotype_file, sep='\t')
    if rank == 0:
        print(f"Phenotype data loaded, shape: {phenotype_data.shape}")
        print(f"Columns: {list(phenotype_data.columns)}")
    
    # Check for NaNs in phenotype data
    if rank == 0:
        print("\nChecking for NaNs in phenotype data:")
    covariate_columns = ['age', 'gender', 'BMI'] + [f'PC{i}' for i in range(1, 7)]
    
    for column in covariate_columns:
        if column in phenotype_data.columns:
            nan_count = phenotype_data[column].isna().sum()
            if rank == 0:
                print(f"Column '{column}': {nan_count} NaN values ({nan_count/len(phenotype_data):.2%})")
            
            # If NaNs exist, fill them
            if nan_count > 0:
                if column == 'gender':
                    # For categorical, use mode
                    fill_value = phenotype_data[column].mode()[0]
                else:
                    # For numerical, use mean
                    fill_value = phenotype_data[column].mean()
                
                if rank == 0:
                    print(f"  Filling NaNs with {fill_value}")
                phenotype_data[column] = phenotype_data[column].fillna(fill_value)
    
    # Discover subjects from phenotype file that have complete chromosome data
    # This is much faster than discovering all subjects first!
    if rank == 0:
        filtered_subjects = discover_subjects_from_phenotype(phenotype_data, rank=rank)
    else:
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            filtered_subjects = discover_subjects_from_phenotype(phenotype_data, rank=rank)
        finally:
            sys.stdout = old_stdout
    
    if len(filtered_subjects) == 0:
        raise ValueError("No subjects have both genotype and phenotype data!")
    
    # Get input size from first subject
    first_subject = filtered_subjects[0]
    if rank == 0:
        input_size = get_input_size_from_subject(first_subject[0], first_subject[1])
    else:
        # Non-rank-0: compute silently
        genotype_data = load_genotype_data_for_subject(first_subject[0], first_subject[1])
        input_size = genotype_data.shape[0]

    # STRATIFICATION LOGIC
    is_multilabel = len(disease_labels) > 1
    stratify_labels = None

    if not use_rotation:
        stratify_labels = []

        for id1, id2 in filtered_subjects:
            pheno = phenotype_data.loc[
                (phenotype_data["ID_1"].astype(str) == str(id1)) &
                (phenotype_data["ID_2"].astype(str) == str(id2))
            ]

            if len(pheno) == 0:
                stratify_labels.append(0)
                continue

            if is_multilabel:
                # CASE vs CONTROL
                disease_sum = pheno[disease_labels].values.sum()
                stratify_labels.append(int(disease_sum > 0))
            else:
                # SINGLE LABEL
                stratify_labels.append(int(pheno[disease_labels[0]].values[0]))

        stratify_labels = np.asarray(stratify_labels)

        if rank == 0:
            print("\nStratified split enabled:")
            print(
                "  Mode:",
                "multi-label (case/control)" if is_multilabel else "single-label"
            )
            print(f"  Positive rate: {stratify_labels.mean():.4f}")

    # Split subjects into train/test
    train_subjects, test_subjects = train_test_split(
        filtered_subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    if rank == 0 and stratify_labels is not None:
        def rate(subs):
            idx = [filtered_subjects.index(s) for s in subs]
            return stratify_labels[idx].mean()

        print(f"\nCase/positive rate check:")
        print(f"  Train: {rate(train_subjects):.4f}")
        print(f"  Test : {rate(test_subjects):.4f}")

    return train_subjects, test_subjects, phenotype_data, input_size


def create_dataloaders_with_rotation(train_subjects, test_subjects, phenotype_data, disease_labels,
                                     batch_size=5, num_workers=2,
                                     use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                                     norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                                     use_rotation=True, target_ratio=5, epoch=0, rank=0, apply_snv_filter=True):
    """
    Create training and test dataloaders WITH ROTATION SUPPORT AND SNV-ONLY FILTERING
    
    Args:
        train_subjects: List of training subject tuples
        test_subjects: List of test subject tuples
        phenotype_data: DataFrame with phenotype data
        disease_labels: List of disease column names
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        use_covariates/use_age/use_gender/use_bmi: Which covariates to include
        norm_age/norm_pcs/norm_gender/norm_bmi: Normalization methods
        use_rotation: Whether to use rotation-based sampling (NEW)
        target_ratio: Target controls:cases ratio (NEW)
        epoch: Current epoch number for rotation (NEW)
        rank: Process rank for distributed training (default 0)
        apply_snv_filter: Whether to apply SNV-only filtering using prefiltered indices (default: True)
    
    Returns:
        Dictionary with 'train' and 'test' DataLoaders, rotation_info (or None)
    """
    if rank == 0:
        print("\n" + "=" * 80)
        print("CREATING DATASETS AND DATALOADERS (SNV-ONLY)")
        print("=" * 80)
    
    # First, create full training dataset to fit normalizers
    full_train_dataset = MultilabelGenotypeDataset(
        train_subjects, 
        phenotype_data, 
        disease_labels, 
        use_covariates=use_covariates,
        use_age=use_age,
        use_gender=use_gender,
        use_bmi=use_bmi,
        norm_age=norm_age,
        norm_pcs=norm_pcs,
        norm_gender=norm_gender,
        norm_bmi=norm_bmi,
        fit_normalizers=True,
        normalizers=None,
        apply_snv_filter=apply_snv_filter,
        rank=rank
    )
    
   # verify_id_matching(phenotype_data)
   # verify_covariates(full_train_dataset)
   # verify_labels(full_train_dataset, phenotype_data, disease_labels)
   # verify_genotypes(full_train_dataset)

    # Get the fitted normalizers from training dataset
    fitted_normalizers = full_train_dataset.get_normalizers()
    
    # If using rotation, identify cases and controls
    if use_rotation:
        # Pre-compute case/control indices ONCE and cache
        if not hasattr(create_dataloaders_with_rotation, '_case_control_cache'):
            if rank == 0:
                print("\n" + "="*80)
                print("First epoch: Computing case/control classification (one-time setup)")
                print("="*80)
            
            case_indices, control_indices = precompute_case_control_indices(
                train_subjects, phenotype_data, disease_labels, rank=rank
            )
            
            # Cache the results
            create_dataloaders_with_rotation._case_control_cache = {
                'case_indices': case_indices,
                'control_indices': control_indices
            }
        else:
            if rank == 0:
                print("\nUsing cached case/control indices")
            case_indices = create_dataloaders_with_rotation._case_control_cache['case_indices']
            control_indices = create_dataloaders_with_rotation._case_control_cache['control_indices']
        
        if rank == 0:
            print(f"\nMulti-label class distribution:")
            print(f"  Cases (≥1 disease): {len(case_indices):,}")
            print(f"  Controls (0 diseases): {len(control_indices):,}")
            print(f"  Original ratio: 1:{len(control_indices)/max(1, len(case_indices)):.1f}")
        
        # Create or use existing rotation sampler
        if not hasattr(create_dataloaders_with_rotation, '_rotation_sampler'):
            create_dataloaders_with_rotation._rotation_sampler = RotatingControlSampler(
                case_subject_indices=case_indices,
                control_subject_indices=control_indices,
                target_ratio=target_ratio,
                n_rotations=None,
                random_state=42,
                rank=rank
            )
        
        sampler = create_dataloaders_with_rotation._rotation_sampler
        
        # Get indices for this epoch 
        epoch_indices = sampler.get_epoch_indices(epoch)
        rotation_info = sampler.get_rotation_info(epoch)
        sampler.print_rotation_info(epoch)
        
        # Create subset of train_subjects for this epoch
        epoch_train_subjects = [train_subjects[i] for i in epoch_indices]
        
        # Create epoch-specific dataset
        train_dataset = MultilabelGenotypeDataset(
            epoch_train_subjects, 
            phenotype_data, 
            disease_labels, 
            use_covariates=use_covariates,
            use_age=use_age,
            use_gender=use_gender,
            use_bmi=use_bmi,
            norm_age=norm_age,
            norm_pcs=norm_pcs,
            norm_gender=norm_gender,
            norm_bmi=norm_bmi,
            fit_normalizers=False,
            normalizers=fitted_normalizers,
            apply_snv_filter=apply_snv_filter,
            rank=rank
        )
    else:
        # No rotation - use all training data
        train_dataset = full_train_dataset
        rotation_info = None
    
    # Create test dataset using fitted normalizers (NEVER rotate test set)
    test_dataset = MultilabelGenotypeDataset(
        test_subjects,
        phenotype_data, 
        disease_labels, 
        use_covariates=use_covariates,
        use_age=use_age,
        use_gender=use_gender,
        use_bmi=use_bmi,
        norm_age=norm_age,
        norm_pcs=norm_pcs,
        norm_gender=norm_gender,
        norm_bmi=norm_bmi,
        fit_normalizers=False,
        normalizers=fitted_normalizers,
        apply_snv_filter=apply_snv_filter,
        rank=rank
    )
    
    # Create dataloaders with optimizations
    if rank == 0:
        print("\nCreating DataLoaders...")
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            prefetch_factor=1, 
            persistent_workers=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True, 
            prefetch_factor=1, 
            persistent_workers=True
        )
    }
    
    if rank == 0:
        print("DataLoaders created successfully")
        print(f"- Train batches: {len(dataloaders['train'])}")
        print(f"- Test batches: {len(dataloaders['test'])}")
    
      #  verify_case_control_integrity(train_subjects, test_subjects, phenotype_data, disease_labels)

    return dataloaders, rotation_info


# For backward compatibility - original function without rotation
def create_dataloaders(train_subjects, test_subjects, phenotype_data, disease_labels,
                       batch_size=5, num_workers=2,
                       use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                       norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                       apply_snv_filter=True):
    """
    Original create_dataloaders function (without rotation)
    Kept for backward compatibility
    WITH SNV-ONLY FILTERING SUPPORT
    """
    dataloaders, _ = create_dataloaders_with_rotation(
        train_subjects, test_subjects, phenotype_data, disease_labels,
        batch_size=batch_size, num_workers=num_workers,
        use_covariates=use_covariates, use_age=use_age, use_gender=use_gender, use_bmi=use_bmi,
        norm_age=norm_age, norm_pcs=norm_pcs, norm_gender=norm_gender, norm_bmi=norm_bmi,
        use_rotation=False, target_ratio=5, epoch=0, apply_snv_filter=apply_snv_filter
    )
    return dataloaders, None


# Example usage and testing
if __name__ == "__main__":
    print("Testing updated dataloader with rotation and SNV-only filtering...")
    
    # Disease labels based on new phenotype file columns
    disease_labels = ['PrC', 'PanC', 'CRC', 'BC', 'T2D']
    
    # Prepare data splits
    train_subjects, test_subjects, phenotype_data, input_size = prepare_data_splits(
        disease_labels=disease_labels,
        phenotype_file=PHENOTYPE_FILE,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nInput size (total SNVs): {input_size:,}")
    
    # Test rotation across 3 epochs
    for epoch in range(3):
        print(f"\n\n{'#'*80}")
        print(f"# TESTING EPOCH {epoch} (SNV-ONLY)")
        print(f"{'#'*80}")
        
        # Create dataloaders with rotation
        dataloaders, rotation_info = create_dataloaders_with_rotation(
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            phenotype_data=phenotype_data,
            disease_labels=disease_labels,
            batch_size=2,
            num_workers=2,
            use_covariates=True,
            use_age=True,
            use_gender=True,
            use_bmi=True,
            norm_age="standard",
            norm_pcs="standard",
            norm_gender="none",
            norm_bmi="standard",
            use_rotation=True,
            target_ratio=5,
            epoch=epoch,
            apply_snv_filter=True
        )
        
        # Test loading a batch
        print(f"\nTesting batch loading for epoch {epoch}...")
        train_loader = dataloaders['train']
        for batch_idx, (genotypes, covariates, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Genotypes shape: {genotypes.shape}")
            print(f"  Covariates shape: {covariates.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Genotypes dtype: {genotypes.dtype}")
            print(f"  Genotypes value range: [{genotypes.min():.2f}, {genotypes.max():.2f}]")
            print(f"  Labels: {labels}")
            
            if batch_idx == 0:  # Only test first batch
                break
    
    print("\n" + "=" * 80)
    print("ROTATION DATALOADER TEST WITH SNV-ONLY FILTERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)