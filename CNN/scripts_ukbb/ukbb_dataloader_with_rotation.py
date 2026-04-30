"""
Updated DataLoader for Chromosome-wise uint8 format GWAS data
WITH ROTATION-BASED CONTROL SUBSAMPLING FOR IMBALANCED MULTI-LABEL DATA

New Features:
- RotatingControlSampler for handling severe class imbalance
- Rotates through control samples across epochs
- All controls used over multiple epochs
- Configurable control:case ratios
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

# CHR_1_10_BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8"
# CHR_11_22_BASE_PATH = "/mnt/fast/nobackup/scratch4weeks/if00208/ukbb/samples_chr_wise_uint8"

# Phenotype file path
PHENOTYPE_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/ukb_cancers_t2d_ukb676869_13102025_cleaned.tsv"


# def get_chromosome_directory(chr_num):
#     """
#     Get the appropriate base directory and subdirectory for a given chromosome
    
#     Args:
#         chr_num: Chromosome number (1-22)
    
#     Returns:
#         Full path to chromosome directory
#     """
#     if 1 <= chr_num <= 10:
#         base_path = CHR_1_10_BASE_PATH
#     elif 11 <= chr_num <= 22:
#         base_path = CHR_11_22_BASE_PATH
#     else:
#         raise ValueError(f"Invalid chromosome number: {chr_num}. Must be 1-22.")
    
#     chr_dir = os.path.join(base_path, f"chr{chr_num}")
#     return chr_dir

def precompute_case_control_indices(train_subjects, phenotype_data, disease_labels):
    """
    Efficiently identify case and control indices using vectorized operations
    
    Returns:
        case_indices: List of indices where subject has ≥1 disease
        control_indices: List of indices where subject has 0 diseases
    """
    print("Pre-computing case/control indices (one-time operation)...")
    start_time = time.time()
    
    # Create a mapping dataframe for fast lookup
    # Convert train_subjects to DataFrame
    train_df = pd.DataFrame(train_subjects, columns=['ID_1', 'ID_2'])
    train_df['idx'] = range(len(train_subjects))
    
    # Ensure ID types match
    train_df['ID_1'] = train_df['ID_1'].astype(str)
    train_df['ID_2'] = train_df['ID_2'].astype(str)
    phenotype_data_copy = phenotype_data.copy()
    phenotype_data_copy['ID_1'] = phenotype_data_copy['ID_1'].astype(str)
    phenotype_data_copy['ID_2'] = phenotype_data_copy['ID_2'].astype(str)
    
    # Merge with phenotype data (vectorized operation!)
    merged = train_df.merge(
        phenotype_data_copy[['ID_1', 'ID_2'] + disease_labels],
        on=['ID_1', 'ID_2'],
        how='left'
    )
    
    # Compute sum of diseases for each subject (vectorized!)
    merged['disease_sum'] = merged[disease_labels].sum(axis=1)
    
    # Identify cases and controls (vectorized!)
    case_indices = merged[merged['disease_sum'] > 0]['idx'].tolist()
    control_indices = merged[merged['disease_sum'] == 0]['idx'].tolist()
    
    elapsed = time.time() - start_time
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


def discover_all_subjects():
    """
    Discover all subjects that have complete data across all 22 chromosomes
    
    Returns:
        List of tuples: [(id1, id2), ...]
    """
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
    
    print(f"Found {len(subjects_with_complete_data)} subjects with complete data (all 22 chromosomes)")
    return subjects_with_complete_data


def load_genotype_data_for_subject(id1, id2, return_shapes=False):
    """
    Load and concatenate genotype data from all 22 chromosomes for a subject
    
    Args:
        id1: First ID
        id2: Second ID
        return_shapes: If True, return chromosome shapes for debugging
    
    Returns:
        Concatenated genotype array of shape (total_snps, 3)
        If return_shapes=True, also returns list of shapes per chromosome
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
        
        # Convert uint8 to float32 (more memory efficient than float64)
        chr_data = chr_data.astype(np.float32) / 255
        
        chromosome_data.append(chr_data)
        chromosome_shapes.append(chr_data.shape)
    
    # Concatenate all chromosomes along SNP dimension
    genotype_data = np.concatenate(chromosome_data, axis=0)  # Shape: (total_snps, 3)
    
    if return_shapes:
        return genotype_data, chromosome_shapes
    else:
        return genotype_data


def get_input_size_from_subject(id1, id2):
    """
    Get the total number of SNPs for a subject by loading their data
    
    Args:
        id1: First ID
        id2: Second ID
    
    Returns:
        Total number of SNPs across all chromosomes
    """
    genotype_data, shapes = load_genotype_data_for_subject(id1, id2, return_shapes=True)
    total_snps = genotype_data.shape[0]
    
    print(f"\nSNP counts per chromosome for subject {id1}_{id2}:")
    for chr_num, shape in enumerate(shapes, 1):
        print(f"  Chr {chr_num}: {shape[0]:,} SNPs")
    print(f"  Total: {total_snps:,} SNPs")
    
    return total_snps


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
                 target_ratio=5, n_rotations=None, random_state=42, stratify_by=None):
        """
        Args:
            case_subject_indices: List of indices for case subjects
            control_subject_indices: List of indices for control subjects
            target_ratio: Desired controls:cases ratio (e.g., 5 means 5 controls per 1 case)
            n_rotations: Number of control subsets (auto-calculated if None)
            random_state: Random seed for reproducibility
            stratify_by: Optional dict mapping subject indices to strata (e.g., age groups)
        """
        self.case_indices = np.array(case_subject_indices)
        self.control_indices = np.array(control_subject_indices)
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.stratify_by = stratify_by
        
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
        """Print formatted rotation information"""
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
    """
    
    def __init__(self, subject_list, phenotype_data, disease_labels, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                 norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                 fit_normalizers=True, normalizers=None):
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
        """
        self.subject_list = subject_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        
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
            genotype_tensor: Shape (n_snps, 3)
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
        
        # Load genotype data
        genotype_data = load_genotype_data_for_subject(id1, id2)
        genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor


def prepare_data_splits(disease_labels, test_size=0.2, random_state=42):
    """
    Prepare train/test splits for the dataset
    
    Args:
        disease_labels: List of disease column names
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, test_dataset, dataloaders, input_size, phenotype_data
    """
    print("=" * 80)
    print("PREPARING DATA SPLITS")
    print("=" * 80)
    
    # Load phenotype data
    print(f"\nLoading phenotype data from: {PHENOTYPE_FILE}")
    phenotype_data = pd.read_csv(PHENOTYPE_FILE, sep='\t')
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")
    print(f"Columns: {list(phenotype_data.columns)}")
    
    # Check for NaNs in phenotype data
    print("\nChecking for NaNs in phenotype data:")
    covariate_columns = ['age', 'gender', 'BMI'] + [f'PC{i}' for i in range(1, 7)]
    
    for column in covariate_columns:
        if column in phenotype_data.columns:
            nan_count = phenotype_data[column].isna().sum()
            print(f"Column '{column}': {nan_count} NaN values ({nan_count/len(phenotype_data):.2%})")
            
            # If NaNs exist, fill them
            if nan_count > 0:
                if column == 'gender':
                    # For categorical, use mode
                    fill_value = phenotype_data[column].mode()[0]
                else:
                    # For numerical, use mean
                    fill_value = phenotype_data[column].mean()
                    
                print(f"  Filling NaNs with {fill_value}")
                phenotype_data[column] = phenotype_data[column].fillna(fill_value)
    
    # Discover all subjects with complete chromosome data
    all_subjects = discover_all_subjects()
    
    if len(all_subjects) == 0:
        raise ValueError("No subjects found with complete chromosome data!")
    
    # Filter subjects to only those present in phenotype data
    phenotype_ids = set(phenotype_data['ID_1'].astype(str).unique())
    
    filtered_subjects = []
    for id1, id2 in all_subjects:
        if str(id1) in phenotype_ids:
            filtered_subjects.append((id1, id2))
    
    print(f"\nSubjects with both genotype and phenotype data: {len(filtered_subjects)}")
    
    if len(filtered_subjects) == 0:
        raise ValueError("No subjects have both genotype and phenotype data!")
    
    # Get input size from first subject
    first_subject = filtered_subjects[0]
    input_size = get_input_size_from_subject(first_subject[0], first_subject[1])
    
    # Split subjects into train/test
    train_subjects, test_subjects = train_test_split(
        filtered_subjects, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\nData split: Train {len(train_subjects)}, Test {len(test_subjects)}")
    
    return train_subjects, test_subjects, phenotype_data, input_size


def create_dataloaders_with_rotation(train_subjects, test_subjects, phenotype_data, disease_labels,
                                     batch_size=5, num_workers=4,
                                     use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                                     norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                                     use_rotation=True, target_ratio=5, epoch=0):
    """
    Create training and test dataloaders WITH ROTATION SUPPORT
    
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
    
    Returns:
        Dictionary with 'train' and 'test' DataLoaders, rotation_info (or None)
    """
    print("\n" + "=" * 80)
    print("CREATING DATASETS AND DATALOADERS")
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
        normalizers=None
    )
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = full_train_dataset.get_normalizers()
    
    # If using rotation, identify cases and controls
    if use_rotation:
        # Pre-compute case/control indices ONCE and cache
        if not hasattr(create_dataloaders_with_rotation, '_case_control_cache'):
            print("\n" + "="*80)
            print("First epoch: Computing case/control classification (one-time setup)")
            print("="*80)
            
            case_indices, control_indices = precompute_case_control_indices(
                train_subjects, phenotype_data, disease_labels
            )
            
            # Cache the results
            create_dataloaders_with_rotation._case_control_cache = {
                'case_indices': case_indices,
                'control_indices': control_indices
            }
        else:
            print("\nUsing cached case/control indices (instant!)")
            case_indices = create_dataloaders_with_rotation._case_control_cache['case_indices']
            control_indices = create_dataloaders_with_rotation._case_control_cache['control_indices']
        
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
                random_state=42
            )
        
        sampler = create_dataloaders_with_rotation._rotation_sampler
        
        # Get indices for this epoch (fast!)
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
            normalizers=fitted_normalizers
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
        normalizers=fitted_normalizers
    )
    
    # Create dataloaders with optimizations
    print("\nCreating DataLoaders...")
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            prefetch_factor=2, 
            persistent_workers=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True, 
            prefetch_factor=2, 
            persistent_workers=True
        )
    }
    
    print("DataLoaders created successfully")
    print(f"- Train batches: {len(dataloaders['train'])}")
    print(f"- Test batches: {len(dataloaders['test'])}")
    
    return dataloaders, rotation_info


# For backward compatibility - original function without rotation
def create_dataloaders(train_subjects, test_subjects, phenotype_data, disease_labels,
                       batch_size=5, num_workers=4,
                       use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                       norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard"):
    """
    Original create_dataloaders function (without rotation)
    Kept for backward compatibility
    """
    dataloaders, _ = create_dataloaders_with_rotation(
        train_subjects, test_subjects, phenotype_data, disease_labels,
        batch_size=batch_size, num_workers=num_workers,
        use_covariates=use_covariates, use_age=use_age, use_gender=use_gender, use_bmi=use_bmi,
        norm_age=norm_age, norm_pcs=norm_pcs, norm_gender=norm_gender, norm_bmi=norm_bmi,
        use_rotation=False, target_ratio=5, epoch=0
    )
    return dataloaders, None


# Example usage and testing
if __name__ == "__main__":
    print("Testing updated dataloader with rotation...")
    
    # Disease labels based on new phenotype file columns
    disease_labels = ['PrC', 'PanC', 'CRC', 'BC', 'T2D']
    
    # Prepare data splits
    train_subjects, test_subjects, phenotype_data, input_size = prepare_data_splits(
        disease_labels=disease_labels,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nInput size (total SNPs): {input_size:,}")
    
    # Test rotation across 3 epochs
    for epoch in range(3):
        print(f"\n\n{'#'*80}")
        print(f"# TESTING EPOCH {epoch}")
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
            epoch=epoch
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
    print("ROTATION DATALOADER TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)