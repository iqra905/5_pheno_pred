import os
import numpy as np
import pandas as pd
import gzip
import glob
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute PCA features for SNP data")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d', help="Directory containing genotype files")
    parser.add_argument("-output_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_pca', help="Directory to save PCA features")
    parser.add_argument("-n_components", type=int, default=49,  help="Number of PCA components")
    parser.add_argument("-test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("-random_seed", type=int, default=42, help="Random seed for train-test split")
    parser.add_argument("-n_samples", type=int, default=1000,  help="Number of samples to use for fitting PCA")
    parser.add_argument("-workers", type=int, default=10, help="Number of worker processes for parallel computation")
    return parser.parse_args()

def load_and_convert_to_dosage(file_path):
    """Load a genotype file and convert to dosage encoding"""
    try:
        with gzip.open(file_path, 'rt') as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)
        
        # Convert to dosage encoding (0*AA + 1*AB + 2*BB)
        data_values = data.values
        dosage = data_values[:, 0] * 0 + data_values[:, 1] * 1 + data_values[:, 2] * 2
        
        return dosage
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_sample_id_from_path(file_path):
    """Extract sample ID from file path"""
    return os.path.basename(file_path).replace("sample_", "").replace(".gen.gz", "")

def process_and_save_file(file_path, output_dir, pca):
    """Process a single file and save PCA features"""
    sample_id = get_sample_id_from_path(file_path)
    output_file = os.path.join(output_dir, f"sample_{sample_id}_pca.npy")
    
    # Skip if already processed
    if os.path.exists(output_file):
        return None
    
    # Load and convert to dosage
    dosage = load_and_convert_to_dosage(file_path)
    if dosage is None:
        return None
    
    # Apply PCA transformation
    dosage = dosage.reshape(1, -1)
    pca_features = pca.transform(dosage)
    
    # Save PCA features
    np.save(output_file, pca_features)
    return sample_id

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of genotype files
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
    file_list.sort()
    print(f"Found {len(file_list)} genotype files")
    
    # # Select subset for PCA fitting
    # np.random.seed(42)
    # #n_samples = min(args.n_samples, len(file_list))
    # n_samples = len(file_list)
    # pca_fitting_files = np.random.choice(file_list, n_samples, replace=False)

    # Perform train-test split with the same parameters as the training script
    train_files, test_files = train_test_split(
        file_list, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Split data: {len(train_files)} training files, {len(test_files)} test files")
    
    # Save train-test split for reference
    with open(os.path.join(args.output_dir, "train_files.txt"), 'w') as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")
    with open(os.path.join(args.output_dir, "test_files.txt"), 'w') as f:
        for file_path in test_files:
            f.write(f"{file_path}\n")
    
    # Select subset of training files for PCA fitting
    np.random.seed(args.random_seed)
    n_samples = min(args.n_samples, len(train_files))
    #n_samples = len(train_files)
    pca_fitting_files = np.random.choice(train_files, n_samples, replace=False)
    
    # Load and convert dosage data for PCA fitting
    print(f"Loading {n_samples} samples for PCA fitting...")
    all_dosages = []
    for i, file in enumerate(tqdm(pca_fitting_files)):
        dosage = load_and_convert_to_dosage(file)
        if dosage is not None:
            all_dosages.append(dosage)
    
    # Stack all dosage data
    print("Stacking all dosage data...")
    stacked_dosages = np.vstack(all_dosages)
    
    # Fit PCA
    print(f"Fitting PCA with {args.n_components} components...")
    n_components = min(args.n_components, stacked_dosages.shape[0] - 1)
    if n_components < args.n_components:
        print(f"Warning: Requested {args.n_components} components, but can only use {n_components} due to data constraints")
    
    pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
    pca.fit(stacked_dosages)
    
    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA fitted. Total explained variance: {explained_var:.2f}%")
    if len(pca.explained_variance_ratio_) > 5:
        top5_variance = pca.explained_variance_ratio_[:5].sum() * 100
        print(f"Top 5 components explain {top5_variance:.2f}% of variance")
    
    # Save the PCA model
    pca_model_path = os.path.join(args.output_dir, "pca_model.pkl")
    import pickle
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_model_path}")
    
    # Process and save PCA features for all files
    print(f"Processing all {len(file_list)} files and saving PCA features...")
    
    # Use multiprocessing for faster processing
    process_fn = partial(process_and_save_file, output_dir=args.output_dir, pca=pca)
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(process_fn, file_list), total=len(file_list)))
    
    # Count successful transformations
    processed = [r for r in results if r is not None]
    print(f"Successfully processed and saved PCA features for {len(processed)} files")
    print(f"PCA features saved to {args.output_dir}")

if __name__ == "__main__":
    main()