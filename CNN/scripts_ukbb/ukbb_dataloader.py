"""
Updated DataLoader for Chromosome-wise uint8 format GWAS data
Key Changes:
- 22 separate .npy files per subject (one per chromosome)
- uint8 format instead of float16
- Split across two directory paths
- New phenotype file format (TSV with different column names)
- 6 PCs instead of 10
"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer


# Directory paths for chromosomes
CHR_1_10_BASE_PATH = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8"
CHR_11_22_BASE_PATH = "/mnt/fast/nobackup/scratch4weeks/if00208/ukbb/samples_chr_wise_uint8"

# Phenotype file path
PHENOTYPE_FILE = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_cancers_t2d_ukb676869_13102025.tsv"


def get_chromosome_directory(chr_num):
    """
    Get the appropriate base directory and subdirectory for a given chromosome
    
    Args:
        chr_num: Chromosome number (1-22)
    
    Returns:
        Full path to chromosome directory
    """
    if 1 <= chr_num <= 10:
        base_path = CHR_1_10_BASE_PATH
    elif 11 <= chr_num <= 22:
        base_path = CHR_11_22_BASE_PATH
    else:
        raise ValueError(f"Invalid chromosome number: {chr_num}. Must be 1-22.")
    
    chr_dir = os.path.join(base_path, f"chr{chr_num}")
    return chr_dir


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
        chr_dir = get_chromosome_directory(chr_num)
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
    chr1_dir = get_chromosome_directory(1)
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
        chr_data = chr_data.astype(np.float32)
        
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


class MultilabelGenotypeDataset(Dataset):
    """
    Dataset for chromosome-wise GWAS data with multi-label disease prediction
    
    NEW FORMAT:
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
            fit_normalizers: If True, fit normalizers on this data (for training set)
            normalizers: Pre-fitted normalizers (for test set)
        """
        self.subject_list = subject_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi

        # Verify that all disease label columns exist in the phenotype data
        for label in self.disease_labels:
            if label not in self.phenotype_data.columns:
                raise ValueError(f"Disease label column '{label}' not found in phenotype data. "
                               f"Available columns are: {', '.join(self.phenotype_data.columns)}")

        # Set up normalizers
        if normalizers is None:
            self.age_normalizer = CovariateNormalizer(norm_age)
            self.pcs_normalizer = CovariateNormalizer(norm_pcs)
            self.gender_normalizer = CovariateNormalizer(norm_gender)
            self.bmi_normalizer = CovariateNormalizer(norm_bmi)

            # Fit normalizers if this is training set
            if fit_normalizers:
                self._fit_normalizers()
        else:
            # Use the provided normalizers for test data
            self.age_normalizer = normalizers['age']
            self.pcs_normalizer = normalizers['pcs']
            self.gender_normalizer = normalizers['gender']
            self.bmi_normalizer = normalizers['bmi']
        
        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of subjects: {len(subject_list)}")
        print(f"- Disease labels: {', '.join(disease_labels)}")
        print(f"- Using PCs (6 PCs): {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")
        print(f"- Using BMI: {use_bmi} (normalization: {norm_bmi})")

        # Print disease prevalence information
        self._print_disease_statistics()

        # Print sample information for the first few samples
        self._print_sample_examples(3)

    def _print_sample_examples(self, num_samples=3):
        """Print detailed information about the first few samples in the dataset"""
        print(f"\nSample Examples (first {num_samples}):")
        for i in range(min(num_samples, len(self.subject_list))):
            id1, id2 = self.subject_list[i]
            
            # Get labels for all diseases for this sample
            disease_status = {}
            sample_row = self.phenotype_data[self.phenotype_data['ID_1'] == id1]
            
            if not sample_row.empty:
                for disease in self.disease_labels:
                    status = sample_row[disease].values[0]
                    disease_status[disease] = int(status)
                
                # Include demographic info if available
                demographics = []
                if 'gender' in self.phenotype_data.columns and self.use_gender:
                    gender = sample_row['gender'].values[0]
                    demographics.append(f"Gender: {gender}")
                if 'age' in self.phenotype_data.columns and self.use_age:
                    age = sample_row['age'].values[0]
                    demographics.append(f"Age: {age}")
                if 'BMI' in self.phenotype_data.columns and self.use_bmi:
                    bmi = sample_row['BMI'].values[0]
                    demographics.append(f"BMI: {bmi}")
                
                demo_str = ", ".join(demographics)
                if demo_str:
                    demo_str = f" ({demo_str})"
                
                print(f"Subject ID: {id1}_{id2}{demo_str}")
                print(f"  Disease status: {disease_status}")
            else:
                print(f"Subject ID: {id1}_{id2} - No matching phenotype data found")

    def _print_disease_statistics(self):
        """Print statistics about disease prevalence in the dataset"""
        # Get the sample IDs in this split
        id1_list = [id1 for id1, id2 in self.subject_list]
        
        # Filter phenotype data to only include samples in this split
        split_phenotype = self.phenotype_data[self.phenotype_data['ID_1'].isin(id1_list)]
        
        print("\nDisease Prevalence Statistics:")
        for disease in self.disease_labels:
            if disease in self.phenotype_data.columns:
                count = split_phenotype[disease].sum()
                total = len(split_phenotype)
                percentage = (count / total) * 100
                print(f"- {disease}: {count}/{total} ({percentage:.2f}%)")

    def _fit_normalizers(self):
        """Fit all normalizers on training data"""
        # Get ID_1 list for filtering
        id1_list = [id1 for id1, id2 in self.subject_list]
        train_phenotype = self.phenotype_data[self.phenotype_data['ID_1'].isin(id1_list)]
        
        if self.use_covariates:
            # Get all PC data as a matrix (n_samples, 6) - NOTE: Only 6 PCs now
            pc_data = np.array([train_phenotype[f'PC{i}'].values for i in range(1, 7)]).T
            self.pcs_normalizer.fit(pc_data)
        
        if self.use_age:
            age_data = train_phenotype['age'].values
            self.age_normalizer.fit(age_data)
        
        if self.use_gender:
            gender_data = train_phenotype['gender'].values
            self.gender_normalizer.fit(gender_data)
        
        if self.use_bmi:
            bmi_data = train_phenotype['BMI'].values
            self.bmi_normalizer.fit(bmi_data)
    
    def get_normalizers(self):
        """Return the fitted normalizers"""
        return {
            'age': self.age_normalizer,
            'pcs': self.pcs_normalizer,
            'gender': self.gender_normalizer,
            'bmi': self.bmi_normalizer,
        }
    
    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        """
        Load data for a single subject
        
        Returns:
            genotype_tensor: Shape (total_snps, 3) - concatenated across all 22 chromosomes
            covariates_tensor: Shape (n_covariates,) - up to 9 features (6 PCs + age + gender + BMI)
            labels_tensor: Shape (n_diseases,) - binary labels for each disease
        """
        id1, id2 = self.subject_list[idx]
        
        # Get labels for all diseases
        labels = self.phenotype_data.loc[
            self.phenotype_data['ID_1'] == id1, 
            self.disease_labels
        ].values[0]
        
        # Process covariates with normalization if needed
        covariates_list = []
        
        # PCs (6 PCs instead of 10)
        if self.use_covariates:
            # Get PC values as matrix (1, 6)
            pc_data = np.array([
                self.phenotype_data.loc[self.phenotype_data['ID_1'] == id1, f'PC{i}'].values[0] 
                for i in range(1, 7)  # Changed from range(1, 11) to range(1, 7)
            ]).reshape(1, -1)
            
            # Transform PCs
            normalized_pcs = self.pcs_normalizer.transform(pc_data).flatten()
            covariates_list.append(normalized_pcs)
                
        # Age
        if self.use_age:
            age = self.phenotype_data.loc[self.phenotype_data['ID_1'] == id1, 'age'].values[0]
            normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        # Gender
        if self.use_gender:
            gender = self.phenotype_data.loc[self.phenotype_data['ID_1'] == id1, 'gender'].values[0]
            normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)

        # BMI
        if self.use_bmi:
            bmi = self.phenotype_data.loc[self.phenotype_data['ID_1'] == id1, 'BMI'].values[0]
            normalized_bmi = self.bmi_normalizer.transform(np.array([[bmi]])).flatten()
            covariates_list.append(normalized_bmi)
        
        # Combine all covariates
        covariates = np.concatenate(covariates_list) if covariates_list else np.array([])
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        # Load genotype data from all 22 chromosomes
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


def create_dataloaders(train_subjects, test_subjects, phenotype_data, disease_labels,
                       batch_size=5, num_workers=4,
                       use_covariates=True, use_age=True, use_gender=True, use_bmi=True,
                       norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard"):
    """
    Create training and test dataloaders
    
    Args:
        train_subjects: List of training subject tuples
        test_subjects: List of test subject tuples
        phenotype_data: DataFrame with phenotype data
        disease_labels: List of disease column names
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_covariates/use_age/use_gender/use_bmi: Which covariates to include
        norm_age/norm_pcs/norm_gender/norm_bmi: Normalization methods
    
    Returns:
        Dictionary with 'train' and 'test' DataLoaders
    """
    print("\n" + "=" * 80)
    print("CREATING DATASETS AND DATALOADERS")
    print("=" * 80)
    
    # Create training dataset and fit normalizers
    train_dataset = MultilabelGenotypeDataset(
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
    fitted_normalizers = train_dataset.get_normalizers()
    
    # Create test dataset using fitted normalizers
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
    
    return dataloaders, fitted_normalizers


# Example usage and testing
if __name__ == "__main__":
    print("Testing updated dataloader...")
    
    # Disease labels based on new phenotype file columns
    disease_labels = ['CRC', 'BC', 'PrC', 'PanC', 'T2D']
    
    # Prepare data splits
    train_subjects, test_subjects, phenotype_data, input_size = prepare_data_splits(
        disease_labels=disease_labels,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nInput size (total SNPs): {input_size:,}")
    
    # Create dataloaders
    dataloaders, fitted_normalizers = create_dataloaders(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        phenotype_data=phenotype_data,
        disease_labels=disease_labels,
        batch_size=2,  # Small batch size for testing
        num_workers=2,
        use_covariates=True,
        use_age=True,
        use_gender=True,
        use_bmi=True,
        norm_age="standard",
        norm_pcs="standard",
        norm_gender="none",
        norm_bmi="standard"
    )
    
    # Test loading a batch
    print("\n" + "=" * 80)
    print("TESTING BATCH LOADING")
    print("=" * 80)
    
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
    print("DATALOADER TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)