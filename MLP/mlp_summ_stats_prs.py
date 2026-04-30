import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gzip
import glob
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import multiprocessing as mp


# [All model definitions remain the same]
class SNPsOnlyModel(nn.Module):
    """Model for SNPs-only prediction"""
    def __init__(self, input_size, hidden_sizes, dropout_rate, act):
        super(SNPsOnlyModel, self).__init__()
        
        # For raw SNP data with shape (n_snps, 3), first apply a pointwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        # After conv, shape becomes (n_snps, 1), so we flatten it
        self.flattened_size = input_size  # This is n_snps
        
        # Create main layers
        layers = []
        current_size = self.flattened_size
        
        # Hidden layers
        for i in range(len(hidden_sizes)):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_sizes[i])),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_sizes[i])),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_sizes[i]
        
        # Output layer
        layers.append(('output', nn.Linear(current_size, 1)))
        
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # Reshape input from [batch_size, n_snps, 3] to [batch_size, 3, n_snps]
        x = x.permute(0, 2, 1)
        # Apply pointwise convolution to get [batch_size, 1, n_snps]
        x = self.pointwise_conv(x)
        # Flatten to [batch_size, n_snps]
        x = x.squeeze(1)
        return self.model(x).squeeze(1)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation function {name} not implemented.")


class GenotypeModel(nn.Module):
    """Combined model for both SNPs and covariates"""
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, use_covariates=True, use_age=True, use_gender=True, num_covariates=10):
        super(GenotypeModel, self).__init__()
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender

        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender

        # For raw SNP data with shape (n_snps, 3), apply pointwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        
        # Create main layers
        layers = []
        current_size = input_size
        
        # All hidden layers except the last one
        for i in range(len(hidden_sizes)):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_sizes[i])),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_sizes[i])),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_sizes[i]
        
        self.main_layers = nn.Sequential(OrderedDict(layers))
        
        # Last hidden layer (will concatenate covariates to its output)
        last_hidden_size = hidden_sizes[-1]
        concat_size = last_hidden_size + self.total_covariates
        
        # Process concatenated features
        self.final_processing = nn.Sequential(OrderedDict([
            ('final_linear', nn.Linear(concat_size, last_hidden_size)),
            ('final_bn', nn.BatchNorm1d(last_hidden_size)),
            ('final_activation', self.get_activation(act)),
            ('final_dropout', nn.Dropout(dropout_rate))
        ]))
        
        # Output layer
        self.output_layer = nn.Linear(last_hidden_size, 1)

    def forward(self, x, covariates=None):
        # Reshape input from [batch_size, n_snps, 3] to [batch_size, 3, n_snps]
        x = x.permute(0, 2, 1)
        # Apply pointwise convolution to get [batch_size, 1, n_snps]
        x = self.pointwise_conv(x).squeeze(1)  # -> [batch_size, n_snps]
        
        # Main layers
        x = self.main_layers(x)
        
        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Process concatenated features
        x = self.final_processing(x)
        
        # Output layer
        return self.output_layer(x).squeeze(1)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation function {name} not implemented.")


def get_input_size(genotype_file):
    """Determine the number of SNPs from a genotype file"""
    with gzip.open(genotype_file, 'rt') as f:
        first_line = next(f)
        values = first_line.strip().split()
        if len(values) == 3:  # If format is 3 columns (AA, AB, BB probabilities)
            return sum(1 for _ in f) + 1  # Add 1 for the first line we already read
    
    # If we reach here, reopen the file and count lines
    with gzip.open(genotype_file, 'rt') as f:
        return sum(1 for _ in f)

def extract_snp_effects(model, input_size, device='cuda'):
    """
    Extract SNP effect sizes from the trained MLP model using both
    convolution and first MLP layer weights.
    """
    model = model.to(device)
    model.eval()
    
    if isinstance(model, SNPsOnlyModel):
        # First, get the convolution weights
        conv_weights = model.pointwise_conv.weight.data.squeeze().cpu().numpy()
        print(f"Extracted weights from pointwise convolution. Shape: {conv_weights.shape}")
        
        # Get first linear layer weights
        first_linear = None
        for name, module in model.model.named_children():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        
        if first_linear is None:
            raise ValueError("Could not find first linear layer")
        
        # Extract weights from first linear layer - shape should be [hidden_size, input_size]
        mlp_weights = first_linear.weight.data.cpu().numpy()
        print(f"Extracted weights from first MLP layer. Shape: {mlp_weights.shape}")
        
        # For each SNP, calculate its effect by combining convolution and MLP weights
        effect_sizes = np.zeros(input_size)
        
        
        # Calculate how each convolution weight affects each dosage level
        linear_model = np.array([0, 1, 2])  # Dosage of alt allele
            
        # For each SNP position
        for i in range(input_size):
            # Get importance from first linear layer
            # Look at raw weights without taking absolute values
            snp_weights = mlp_weights[:, i]
            snp_importance = np.mean(snp_weights)  # Can be positive or negative
            
            # Scale by magnitude of weights (overall importance)
            magnitude = np.abs(snp_weights).mean()
            
            # Final effect combines sign and magnitude
            effect_sizes[i] = snp_importance * magnitude
        
        # Normalize to realistic range (standard deviation of 1)
        if np.std(effect_sizes) > 0:
            effect_sizes = effect_sizes / np.std(effect_sizes)
        
        print(f"Created SNP-specific effects using first MLP layer weights")
        return effect_sizes
    
    # Handle other model types here...
    else:
        raise ValueError("Cannot extract SNP effects from this model type")

def process_genotype_file(args):
    """Worker function for processing a single genotype file"""
    gen_file, snp_effects = args
    
    sample_id_str = os.path.basename(gen_file).replace("sample_", "").replace(".gen.gz", "")
    sample_id = int(sample_id_str) if sample_id_str.isdigit() else sample_id_str
    
    try:
        # For raw SNP data
        with gzip.open(gen_file, 'rt') as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)
        
        # Calculate PRS as weighted sum of dosages
        prs = 0.0
        num_snps = min(len(data), len(snp_effects))
        
        for i in range(num_snps):
            probs = data.iloc[i].values
            
            if len(probs) == 3:  # If we have 3 probabilities (AA, AB, BB)
                # Calculate alt allele dosage
                #dosage = probs[1] + 2 * probs[2]  # 1*P(AB) + 2*P(BB)
                dosage = 2 * probs[0] + probs[1]   # 2*P(AA) + 1*P(AB) 
                
                # Accumulate PRS using the effect size from the model
                prs += dosage * snp_effects[i]
        
        return {
            'sample_id': sample_id,
            'mlp_prs': float(prs)
        }
        
    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        return None

def calculate_mlp_prs(genotype_files, snp_effects, output_file, num_processes=None):
    """
    Calculate PRS for samples using effect sizes derived from an MLP model,
    using multiprocessing for parallel execution.
    
    Args:
        genotype_files: List of genotype files
        snp_effects: Effect sizes extracted from MLP model
        output_file: Path to save PRS results
        num_processes: Number of processes to use (default: None, uses CPU count)
        
    Returns:
        DataFrame with sample_id and mlp_prs columns
    """
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count() - 1  # Leave one core free
        num_processes = max(1, num_processes)  # Use at least one core
    
    print(f"Using {num_processes} processes for parallel computation")
    
    # Prepare arguments for each worker
    args_list = [(gen_file, snp_effects) for gen_file in genotype_files]
    
    # Create a pool of workers and map the function to the arguments
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_genotype_file, args_list),
            total=len(args_list),
            desc="Calculating MLP-based PRS"
        ))
    
    # Filter out None results (errors) and create DataFrame
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)
    
    # Sort by sample_id
    results_df = results_df.sort_values('sample_id')
    print("Sorted results by sample ID")
    
    # Standardize the PRS (mean=0, sd=1) for better comparability
    if len(results_df) > 1:
        mean_prs = results_df['mlp_prs'].mean()
        std_prs = results_df['mlp_prs'].std()
        results_df['mlp_prs_std'] = (results_df['mlp_prs'] - mean_prs) / std_prs
        print(f"Raw MLP-PRS stats - Mean: {mean_prs:.4f}, SD: {std_prs:.4f}")
    
    # Save to file
    suffix = "_scores"
    output_file_with_suffix = output_file.replace('.csv', f'{suffix}.csv')
    results_df.to_csv(output_file_with_suffix, index=False)

    #results_df.to_csv(output_file, index=False)
    print(f"MLP-based PRS saved to {output_file}")
    
    return results_df


def evaluate_prs(prs_df, phenotype_data, label_col, output_dir):
    """
    Evaluate the performance of the PRS model
    
    Args:
        prs_df: DataFrame with PRS results
        phenotype_data: DataFrame with phenotype data
        label_col: Column name in phenotype data to use as label
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Map sample IDs to labels
    prs_df['sample_id'] = prs_df['sample_id'].astype(str)
    
    # Convert to integer for matching with phenotype data
    prs_df['sample_id_int'] = prs_df['sample_id'].apply(lambda x: int(x) if x.isdigit() else None)
    
    # Create a mapping from sample_id to label
    sample_id_label_map = {}
    for _, row in phenotype_data.iterrows():
        sample_id = row['new_order']
        label = row[label_col]
        sample_id_label_map[sample_id] = label
    
    # Add labels to PRS results
    prs_df['true_label'] = prs_df['sample_id_int'].map(sample_id_label_map)
    
    # Drop samples with no label
    valid_samples = prs_df.dropna(subset=['true_label'])
    
    if len(valid_samples) == 0:
        print("No valid samples with labels found")
        return {'auc': None, 'num_samples': 0}
    
    # Use standardized PRS if available
    prs_col = 'mlp_prs_std' if 'mlp_prs_std' in valid_samples.columns else 'mlp_prs'
    
    # Calculate AUC if we have both classes
    if len(valid_samples['true_label'].unique()) > 1:
        auc_score = roc_auc_score(valid_samples['true_label'], valid_samples[prs_col])
        print(f"AUC: {auc_score:.4f}")
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(valid_samples['true_label'], valid_samples[prs_col])
        plt.plot(fpr, tpr, label=f'MLP-PRS (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for MLP-derived PRS')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'mlp_prs_roc.png'))
        plt.close()
        
        # Plot PRS distribution by case/control status
        plt.figure(figsize=(10, 8))
        for label_value in sorted(valid_samples['true_label'].unique()):
            subset = valid_samples[valid_samples['true_label'] == label_value]
            plt.hist(subset[prs_col], alpha=0.5, bins=20, 
                    label=f"{'Case' if label_value == 1 else 'Control'}")
        plt.xlabel('MLP-derived PRS')
        plt.ylabel('Count')
        plt.title(f'MLP-PRS Distribution by Case/Control Status (AUC = {auc_score:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'mlp_prs_distribution.png'))
        plt.close()
        
        # Save stats by class
        stats = {}
        for label in [0, 1]:
            subset = valid_samples[valid_samples['true_label'] == label]
            stats[f"class_{label}_count"] = len(subset)
            stats[f"class_{label}_mean_prs"] = subset[prs_col].mean()
            stats[f"class_{label}_std_prs"] = subset[prs_col].std()
        
        stats['auc'] = auc_score
        stats['num_samples'] = len(valid_samples)
        
        # Save stats to csv
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'mlp_prs_stats.csv'), index=False)
        
        return stats
    else:
        print("Warning: Cannot calculate AUC - need samples from both classes")
        return {'auc': None, 'num_samples': len(valid_samples)}


def main():
    parser = argparse.ArgumentParser(description="Calculate PRS using a trained MLP model")
    
    # Required arguments
    parser.add_argument('-model', default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_stats_summ_prs/01_snps_exponential_decay_none_class_weight_maf_0.15_t2d/final_model.pth', help='Path to the trained model file (.pth)')
    parser.add_argument('-model_type', choices=['snps_only', 'full'], default='snps_only', help='Type of model used (snps_only or full)')
    parser.add_argument('-geno_dir', default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise_summ_stats/t2d/0.05', help='Directory containing genotype files')
    parser.add_argument('-output', default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_stats_summ_prs/01_snps_exponential_decay_none_class_weight_maf_0.15_t2d/prs_scores/prs_t2d.csv', help='Path to output file')
    
    # MLP model architecture parameters (must match the trained model)
    parser.add_argument('-hidden_sizes', type=str, default='128,128,128', help='Hidden layer sizes used in the model (comma-separated)')
    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate used in the model')
    parser.add_argument('-act', type=str, default='gelu', choices=['relu', 'gelu', 'tanh'], help='Activation function used in the model')
    
    # Covariates parameters (for full model)
    parser.add_argument('-use_covariates', type=int, default=0, choices=[0, 1], help='Whether the model used covariates')
    parser.add_argument('-use_age', type=int, default=0, choices=[0, 1], help='Whether the model used age')
    parser.add_argument('-use_gender', type=int, default=0, choices=[0, 1],  help='Whether the model used gender')
    
    # Evaluation parameters (optional)
    parser.add_argument('-phenotype_file', type=str, default=None, help='Path to phenotype file for evaluation')
    parser.add_argument('-label_col', type=str, default=None, help='Column name in phenotype file to use as label')
    
    # Multiprocessing parameter
    parser.add_argument('-num_processes', type=int, default=None, help='Number of processes for parallel computation (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Convert hidden_sizes from string to list of integers
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Check if genotype directory exists
    if not os.path.exists(args.geno_dir):
        print(f"Error: Genotype directory not found: {args.geno_dir}")
        return 1
    
    # Get genotype files
    genotype_files = glob.glob(os.path.join(args.geno_dir, "sample_*.gen.gz"))
    if not genotype_files:
        print(f"Error: No genotype files found in {args.geno_dir}")
        return 1
    
    print(f"Found {len(genotype_files)} genotype files")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get input size from the first genotype file
    first_genotype_file = genotype_files[0]
    input_size = get_input_size(first_genotype_file)
    print(f"Using raw SNP data with {input_size} SNPs")
    
    # Create model with the same architecture as the trained model
    print("Creating model...")
    if args.model_type == 'snps_only':
        model = SNPsOnlyModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act
        )
    else:  # full model
        model = GenotypeModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act,
            use_covariates=bool(args.use_covariates),
            use_age=bool(args.use_age),
            use_gender=bool(args.use_gender),
            num_covariates=10
        )
    
    # Load model weights
    print(f"Loading model weights from {args.model}...")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Extract SNP effects
    print("Extracting SNP effects from model...")
    snp_effects = extract_snp_effects(model, input_size, device)
    
    # Save effect sizes with proper column headers
    effects_file = os.path.splitext(args.output)[0] + '_effects.csv'
    effects_df = pd.DataFrame({'Effect': snp_effects})
    effects_df.index.name = 'SNP'  # Rename the index
    effects_df.to_csv(effects_file)
    print(f"Saved SNP effect sizes to {effects_file}")
    
    # Calculate PRS with multiprocessing
    print("Calculating PRS...")
    prs_df = calculate_mlp_prs(
        genotype_files,
        snp_effects,
        args.output,
        num_processes=args.num_processes
    )
    
    # Evaluate PRS if phenotype file is provided
    if args.phenotype_file and args.label_col:
        print("Evaluating PRS...")
        phenotype_data = pd.read_excel(args.phenotype_file) if args.phenotype_file.endswith('.xlsx') else pd.read_csv(args.phenotype_file)
        
        if args.label_col not in phenotype_data.columns:
            print(f"Warning: Label column '{args.label_col}' not found in phenotype data")
            print(f"Available columns: {', '.join(phenotype_data.columns)}")
        else:
            # Create evaluation directory
            eval_dir = os.path.join(output_dir, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Evaluate PRS
            eval_results = evaluate_prs(prs_df, phenotype_data, args.label_col, eval_dir)
            
            # Compare with statistical PRS if provided
            if args.stat_prs_file:
                print("Comparing with statistical PRS...")
                compare_with_statistical_prs(
                    prs_df,
                    args.stat_prs_file,
                    phenotype_data,
                    args.label_col,
                    eval_dir
                )
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())