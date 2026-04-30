# multiscale/data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import CovariateNormalizer, print_sample_examples, print_disease_statistics


class MultilabelGenotypeDataset(Dataset):
    """Dataset for multilabel genotype prediction"""
    
    def __init__(self, file_list, phenotype_data, disease_labels, use_covariates=True, use_age=True, 
                 use_gender=True, use_bmi=True, norm_age="standard", norm_pcs="standard", 
                 norm_gender="none", norm_bmi="standard", fit_normalizers=True, normalizers=None):
        
        self.file_list = file_list
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

        if normalizers is None:
            self.age_normalizer = CovariateNormalizer(norm_age)
            self.pcs_normalizer = CovariateNormalizer(norm_pcs)
            self.gender_normalizer = CovariateNormalizer(norm_gender)
            self.bmi_normalizer = CovariateNormalizer(norm_bmi)

            # Fit normalizers if this is training set
            if fit_normalizers:
                self._fit_normalizers()
        else:
            # Use the provided normalizers: for Test data
            self.age_normalizer = normalizers['age']
            self.pcs_normalizer = normalizers['pcs']
            self.gender_normalizer = normalizers['gender']
            self.bmi_normalizer = normalizers['bmi']
        
        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of files: {len(file_list)}")
        print(f"- Disease labels: {', '.join(disease_labels)}")
        print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")
        print(f"- Using BMI: {use_bmi} (normalization: {norm_bmi})")

        # Print disease prevalence information
        print_disease_statistics(file_list, phenotype_data, disease_labels)

        # Print sample information for the first few samples
        print_sample_examples(file_list, phenotype_data, disease_labels, 3)

    def _fit_normalizers(self):
        """Fit all normalizers on training data"""
        if self.use_covariates:
            # Get all PC data as a matrix (n_samples, n_pcs)
            pc_data = np.array([self.phenotype_data[f'PC{i}'].values for i in range(1, 11)]).T
            self.pcs_normalizer.fit(pc_data)
        
        if self.use_age:
            age_data = self.phenotype_data['Agexit'].values
            self.age_normalizer.fit(age_data)
        
        if self.use_gender:
            gender_data = self.phenotype_data['Sex'].values
            self.gender_normalizer.fit(gender_data)
        
        if self.use_bmi:
            bmi_data = self.phenotype_data['Bmi_C'].values
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
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]

        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".npy", "")
        sample_id = int(sample_id_str)
        
        # Get labels for all diseases
        labels = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.disease_labels].values[0]
        
        # Get covariates if needed
        covariates_list = []

        # PCs
        if self.use_covariates:
            # Get PC values as matrix (1, n_pcs)
            pc_data = np.array([
                self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, f'PC{i}'].values[0] 
                for i in range(1, 11)]).reshape(1, -1)
            
            # Transform PCs
            normalized_pcs = self.pcs_normalizer.transform(pc_data).flatten()
            covariates_list.append(normalized_pcs)
                
        # Age
        if self.use_age:
            # Get and normalize age
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        # Gender
        if self.use_gender:
            # Get and normalize gender
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)

        # BMI
        if self.use_bmi:
            # Get and normalize BMI
            bmi = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Bmi_C'].values[0]
            normalized_bmi = self.bmi_normalizer.transform(np.array([[bmi]])).flatten()
            covariates_list.append(normalized_bmi)
        
        # Combine all covariates
        covariates = np.concatenate(covariates_list) if covariates_list else np.array([])
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5M, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor