# multiscale/data/preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer


class CovariateNormalizer:
    """Normalizer for covariates with multiple normalization methods"""
    
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
        """Fit the scaler on training data"""
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.scaler.fit(data)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return self.scaler.transform(data)
        return data


def get_input_size(genotype_file):
    """Get input size from genotype file"""
    if genotype_file.endswith('.npy'):
        genotype_data = np.load(genotype_file)
        return genotype_data.shape[0]  # Number of SNPs


def print_sample_examples(file_list, phenotype_data, disease_labels, num_samples=3):
    """Print detailed information about the first few samples in the dataset"""
    print(f"\nSample Examples (first {num_samples}):")
    for i in range(min(num_samples, len(file_list))):
        file_path = file_list[i]
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        
        # Get labels for all diseases for this sample
        disease_status = {}
        sample_row = phenotype_data[phenotype_data['new_order'] == sample_id]
        
        if not sample_row.empty:
            for disease in disease_labels:
                status = sample_row[disease].values[0]
                disease_status[disease] = int(status)
            
            # Include demographic info if available
            demographics = []
            if 'Sex' in phenotype_data.columns:
                gender = sample_row['Sex'].values[0]
                demographics.append(f"Gender: {gender}")
            if 'Agexit' in phenotype_data.columns:
                age = sample_row['Agexit'].values[0]
                demographics.append(f"Age: {age}")
            if 'Bmi_C' in phenotype_data.columns:
                bmi = sample_row['Bmi_C'].values[0]
                demographics.append(f"Bmi: {bmi}")
            
            demo_str = ", ".join(demographics)
            if demo_str:
                demo_str = f" ({demo_str})"
            
            print(f"Sample ID: {sample_id}{demo_str}")
            print(f"  Disease status: {disease_status}")
        else:
            print(f"Sample ID: {sample_id} - No matching phenotype data found")


def print_disease_statistics(file_list, phenotype_data, disease_labels):
    """Print statistics about disease prevalence in the dataset"""
    # Get the sample IDs in this split
    sample_ids = []
    for file_path in file_list:
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        sample_ids.append(sample_id)
    
    # Filter phenotype data to only include samples in this split
    split_phenotype = phenotype_data[phenotype_data['new_order'].isin(sample_ids)]
    
    print("\nDisease Prevalence Statistics:")
    for disease in disease_labels:
        if disease in phenotype_data.columns:
            count = split_phenotype[disease].sum()
            total = len(split_phenotype)
            percentage = (count / total) * 100
            print(f"- {disease}: {count}/{total} ({percentage:.2f}%)")


def clean_phenotype_data(phenotype_data):
    """Clean phenotype data by handling NaN values"""
    print("\nChecking for NaNs in phenotype data:")
    for column in ['Agexit', 'Sex', 'Bmi_C'] + [f'PC{i}' for i in range(1, 11)]:
        if column in phenotype_data.columns:
            nan_count = phenotype_data[column].isna().sum()
            print(f"Column '{column}': {nan_count} NaN values ({nan_count/len(phenotype_data):.2%})")
            
            if nan_count > 0:
                if column == 'Sex':
                    fill_value = phenotype_data[column].mode()[0]
                else:
                    fill_value = phenotype_data[column].mean()
                    
                print(f"  Filling NaNs with {fill_value}")
                phenotype_data[column] = phenotype_data[column].fillna(fill_value)
    
    return phenotype_data