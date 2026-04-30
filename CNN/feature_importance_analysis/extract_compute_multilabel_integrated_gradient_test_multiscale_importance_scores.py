import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap
import glob
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import argparse
import sys

# IMPORT MODEL CLASSES FROM TRAINING SCRIPT
try:
    from epic_cnn_model_5D_multilabel_chkpt_npy_multiscale_point_conv_transformer_cls_pool import (
        validate_hardcoded_parameters,
        MultilabelGenotypeModel,
        MultiScaleConvBlock,
        ParallelMultiScaleConvBlock,
        GenomicTransformerBlock,
        DiseaseSpecificAttention,
        SeparateDiseaseHead
    )
    print("✓ Successfully imported model classes from training script")
except ImportError as e:
    print(f"✗ Error importing model classes from training script: {e}")
    print("Please ensure 'epic_cnn_model_5D_multilabel_chkpt_npy_multiscale_point_conv_transformer_cls_pool.py' is in your Python path")
    print("You can add the path using: sys.path.insert(0, '/path/to/training/script/')")
    sys.exit(1)


# ARGUMENT PARSING

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_nested_int_list(s):
    """Parse nested lists like '15,63,255;7,31,127;3,15,63' into [[15,63,255], [7,31,127], [3,15,63]]"""
    if not s or s.lower() == 'none':
        return None
    layers = s.split(';')
    return [[int(x) for x in layer.split(',')] for layer in layers]

def parse_args():
    parser = argparse.ArgumentParser(description="Extract SNPs using gradient-based feature importance (TEST SET ONLY)")
    parser.add_argument("-checkpoint_path", type=str, default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/initial_res/01_snps_5d_ks/checkpoint_epoch_70.pt", help="Path to the trained model checkpoint")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx', help="Path to phenotype file")
    parser.add_argument("-output_dir", type=str, default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/m4/feat_imp/", help="Directory to save the output files")
    parser.add_argument("-batch_size", type=int, default=4, help="Batch size for gradient computation (smaller = less memory)")
    parser.add_argument("-max_samples", type=int, default=200, help="Maximum number of TEST SET samples to use for importance calculation")
    parser.add_argument("-disease_labels", type=str, default="pros01,panca,crc,breacancer,t2dm", help="Comma-separated list of disease labels")
    
    # Data split parameters (must match training script)
    parser.add_argument("-test_size", type=float, default=0.2, help="Test size ratio used during training (must match training script)")
    parser.add_argument("-random_state", type=int, default=42, help="Random state used for train/test split (must match training script)")
    
    # Model architecture parameters
    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[7,3,1], help="Convolution Kernel Size (comma-separated)")
    parser.add_argument("-stride", type=parse_int_list, default=[2,2,1], help="Convolution Stride (comma-separated)")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[8,16,32], help="Convolution channels (comma-separated)")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers (comma-separated)")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu","leakyrelu","rrelu","silu"], help="Activation function for the model")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    
    # Covariate parameters
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_bmi", type=int, default=1, choices=[0, 1], help="Include BMI")
    parser.add_argument("-num_covariates", type=int, default=10,  help="Number of principal components to use")
    
    # Normalization parameters
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender")
    parser.add_argument("-norm_bmi", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="BMI normalization")

    # Multi-scale configuration
    parser.add_argument("-use_multi_scale", type=int, default=1, choices=[0, 1], help="Whether to use multi-scale convolutions (0: no, 1: yes)")
    parser.add_argument("-multi_scale_kernels", type=parse_int_list, default=[15,127], help="Multi-scale kernel sizes for first layer")
    parser.add_argument("-multi_scale_strides", type=parse_int_list, default=[8,16], help="Multi-scale strides for first layer")
    parser.add_argument("-multi_scale_fusion", type=str, default="parallel", choices=["cross_scale", "parallel"], help="Multi-scale fusion strategy: cross_scale (branches see all scales) or parallel (independent branches)")
    parser.add_argument("-multi_scale_mode", type=str, default="hardcoded", choices=["progressive", "hardcoded"], help="Multi-scale mode: 'progressive' (kernel//2^i, stride//2^i) or 'hardcoded' (explicit values for each layer)")
    
    # Hardcoded multi-scale parameters (used when multi_scale_mode="hardcoded")
    parser.add_argument("-hardcoded_kernels", type=parse_nested_int_list, default='16,128,1024;16,64,512;16,32,256', help="Hardcoded kernel sizes for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '15,63,255;7,31,127;3,15,63'")
    parser.add_argument("-hardcoded_strides", type=parse_nested_int_list, default='16,16,16;16,16,16;16,16,16', help="Hardcoded stride values for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '4,16,64;2,8,32;1,4,16'")

    
    # Pointwise convolution
    parser.add_argument("-use_pointwise_conv", type=int, default=0, choices=[0, 1], help="Use pointwise conv")
    parser.add_argument("-pointwise_channels", type=int, default=16, help="Pointwise channels")    

    parser.add_argument("-use_pooling", type=int, default=0, choices=[0, 1], help="Whether to use AdaptiveMaxPool1d after convolution layers (0: no, 1: yes)")
    parser.add_argument("-pool_size", type=int, default=64, help="Size of the adaptive pooling output")
    # Pool Type for all pooling if used
    parser.add_argument("-pool_type", type=str, default="max", choices=["max", "avg"], help="Type of adaptive pooling: 'max' for AdaptiveMaxPool1d, 'avg' for AdaptiveAvgPool1d")
    
    # Transformer parameters
    parser.add_argument("-use_transformer", type=int, default=1, choices=[0, 1], help="Use transformer")
    parser.add_argument("-transformer_layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("-transformer_heads", type=int, default=8, help="Transformer heads")
    parser.add_argument("-transformer_dim", type=int, default=384, help="Transformer dimension")
    parser.add_argument("-transformer_ff_dim", type=int, default=1024, help="Transformer FF dimension")
    parser.add_argument("-transformer_dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("-use_positional_encoding", type=int, default=1, choices=[0, 1], help="Use positional encoding")
    parser.add_argument("-max_seq_len", type=int, default=10000, help="Max sequence length")
    parser.add_argument("-use_cls_token", type=int, default=0, choices=[0, 1], help="Use CLS token")
    parser.add_argument("-use_covariate_tokens", type=int, default=1, choices=[0, 1], help="Use covariate tokens")
    parser.add_argument("-covariate_embed_dim", type=int, default=64, help="Covariate embedding dim")
    parser.add_argument("-covariate_token_strategy", type=str, default="combined", help="Covariate token strategy")
    parser.add_argument("-pooling_strategy", type=str, default="concat", help="Pooling strategy")
    
    # Disease attention
    parser.add_argument("-use_disease_attention", type=int, default=0, choices=[0, 1], help="Whether to use disease-specific attention (0: no, 1: yes)")
    parser.add_argument("-use_separate_heads", type=int, default=0, choices=[0, 1], help="Whether to use separate disease heads (0: no, 1: yes)")
    parser.add_argument("-attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("-attention_dim", type=int, default=256, help="Attention dimension")

    # Feature importance parameters
    parser.add_argument("-sampling_strategy", type=str, default="random", choices=["random", "stratified", "balanced", "first"], help="Strategy for selecting TEST SET samples for importance calculation")
    parser.add_argument("-min_positive_per_disease", type=int, default=50, help="Minimum number of positive cases per disease (for stratified/balanced sampling)")
    
    # Feature importance method options
    parser.add_argument("-importance_method", type=str, default="integrated_gradients", choices=["loss_based", "integrated_gradients"], help="Method for calculating feature importance")
    parser.add_argument("-disease_specific", type=int, default=0, choices=[0, 1], help="Calculate importance only on positive cases (1) or all cases (0)")
    parser.add_argument("-use_class_weights", type=int, default=0, choices=[0, 1], help="Use class weights to handle imbalance")
    parser.add_argument("-ig_steps", type=int, default=75, help="Number of steps for integrated gradients")
    parser.add_argument("-baseline_type", type=str, default="zero", choices=["zero", "shuffle", "population_mean"], help="Type of baseline for integrated gradients")
    parser.add_argument("-debug_gradients", type=int, default=1, choices=[0, 1], help="Enable gradient debugging output")
        
    parser.add_argument("-importance_scope", type=str, default="overall", choices=["disease_wise", "overall"], help="Calculate importance per disease (disease_wise) or across all diseases (overall)")
    
    parser.add_argument("-disease_weights", type=str, default="equal", choices=["equal", "balanced", "custom"], help="How to weight diseases in overall mode (equal weights, balanced by prevalence, or custom)")
    # Custom weights if using custom mode
    parser.add_argument("-custom_weights", type=str, default="1.0,1.0,1.0,1.0,1.0", help="Custom weights for diseases (comma-separated, same order as disease_labels)")
    
    return parser.parse_args()

def get_input_size(genotype_file):
    if genotype_file.endswith('.npy'):
        genotype_data = np.load(genotype_file)
        return genotype_data.shape[0]  # Number of SNPs

class CovariateNormalizer:
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
    
    def state_dict(self):
        """Return state for saving/loading"""
        return {
            'method': self.method, 
            'scaler': self.scaler
            }
    
    def load_state_dict(self, state_dict):
        """Load state from saved checkpoint"""
        self.method = state_dict['method']
        self.scaler = state_dict['scaler']

class SimpleGenotypeDataset(Dataset):
    """Simplified dataset for gradient computation using TEST SET data"""
    def __init__(self, file_list, phenotype_data, disease_labels, normalizers=None, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True, 
                 num_covariates=10):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.num_covariates = num_covariates
        
        # Use provided normalizers (should be fitted on training data)
        if normalizers is None:
            raise ValueError("Normalizers must be provided and should be fitted on training data")
        
        self.normalizers = normalizers
        
        # Open memory-mapped files
        self.file_handles = {}
        for file in file_list:
            f = open(file, 'rb')
            self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".npy", "")
        sample_id = int(sample_id_str)

        # Get labels for all diseases
        row = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id]
        if row.shape[0] == 0:
            raise KeyError(f"Sample id {sample_id} not found in phenotype data")
        labels = row[self.disease_labels].values[0]
        
        # Get covariates if needed
        covariates_list = []
        
        # PCs
        if self.use_covariates:
            # Get PC values as matrix (1, n_pcs)
            pc_data = np.array([
                self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, f'PC{i}'].values[0] 
                for i in range(1, self.num_covariates + 1)]).reshape(1, -1)
            # Transform PCs
            normalized_pcs = self.normalizers['pcs'].transform(pc_data).flatten()
            covariates_list.append(normalized_pcs)
        
        # Age
        if self.use_age:
            # Get and normalize age
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            normalized_age = self.normalizers['age'].transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        # Gender
        if self.use_gender:
            # Get and normalize gender
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            normalized_gender = self.normalizers['gender'].transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)
        
        # BMI
        if self.use_bmi:
            bmi = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Bmi_C'].values[0]
            normalized_bmi = self.normalizers['bmi'].transform(np.array([[bmi]])).flatten()
            covariates_list.append(normalized_bmi)
        
        # Combine all covariates
        covariates = np.concatenate(covariates_list) if covariates_list else np.array([])
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5M, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor


def create_baseline(batch_input, baseline_type="zero", dataset=None):
    """Create baseline for integrated gradients"""
    if baseline_type == "zero":
        print(f"  Baseline is : {baseline_type}")
        return torch.zeros_like(batch_input)
    elif baseline_type == "shuffle":
        # Shuffle along SNP dimension for each sample
        baseline = batch_input.clone()
        print(f"  Baseline is : {baseline_type}")

        for i in range(baseline.shape[0]):
            perm = torch.randperm(baseline.shape[1])
            baseline[i] = baseline[i][perm]
        return baseline
    else:
        print(f"  Baseline is : None")
        return torch.zeros_like(batch_input)

def integrated_gradients(model, batch_input, covariates, target_disease_idx, 
                        baseline=None, steps=50, device='cpu', debug=False):
    """
    Robust version of integrated gradients that handles BatchNorm issues
    """
    if debug:
        print(f"\nDEBUG: Integrated Gradients Robust")
        print(f"  Input shape: {batch_input.shape}")
        print(f"  Model training state: {model.training}")
        print(f"  Device: {device}")
    
    # Store original model state
    was_training = model.training
    
    # Temporarily set model to train mode for gradient computation
    # This is needed because BatchNorm in eval mode can block gradients
    model.train()
    
    # Disable dropout and other training-specific behaviors
    def set_bn_eval(module):
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    
    # Set BatchNorm layers to eval mode while keeping model in train mode
    model.apply(set_bn_eval)
    
    if baseline is None:
        baseline = torch.zeros_like(batch_input)
    
    # Ensure tensors are on correct device
    batch_input = batch_input.to(device)
    baseline = baseline.to(device)
    
    # Initialize gradient accumulator
    integrated_grads = None
    
    # Calculate gradients at each interpolation step
    for step in range(steps):
        alpha = step / (steps - 1) if steps > 1 else 1.0
        
        # Create interpolated input
        interpolated = baseline + alpha * (batch_input - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        
        # Handle covariates
        if covariates is not None and covariates.numel() > 0:
            cov_for_forward = covariates.to(device).detach()
        else:
            cov_for_forward = None
        
        # Forward pass
        outputs = model(interpolated, cov_for_forward)
        
        # Select disease output
        disease_output = outputs[:, target_disease_idx]
        
        # Use autograd.grad for more control
        try:
            grads = torch.autograd.grad(
                outputs=disease_output.sum(),
                inputs=interpolated,
                retain_graph=False,
                create_graph=False,
                allow_unused=False
            )[0]
            
            if integrated_grads is None:
                integrated_grads = grads / steps
            else:
                integrated_grads += grads / steps
            
            if debug and step == 0:
                print(f"  ✓ Gradient computation successful")
                print(f"  Gradient norm: {grads.norm().item():.6f}")
                
        except RuntimeError as e:
            if debug:
                print(f"  autograd.grad failed at step {step}: {e}")
            # Fallback to backward()
            disease_output.sum().backward()
            if interpolated.grad is not None:
                if integrated_grads is None:
                    integrated_grads = interpolated.grad / steps
                else:
                    integrated_grads += interpolated.grad / steps
                interpolated.grad = None
                if debug:
                    print(f"  ✓ Fallback to backward() successful")
    
    # Restore original model state
    model.train(was_training)
    
    if integrated_grads is None:
        print("Warning: No gradients computed, returning zeros")
        return torch.zeros(batch_input.shape[0], batch_input.shape[1]).to(device)
    
    # Multiply by input difference
    integrated_grads *= (batch_input - baseline)
    
    # Average across channels
    importance = integrated_grads.abs().mean(dim=2)
    
    if debug:
        print(f"  Final importance shape: {importance.shape}")
        print(f"  Importance range: [{importance.min().item():.6f}, {importance.max().item():.6f}]")
    
    return importance

def calculate_feature_importance(model, dataloader, disease_labels, device, args):
    """
    Calculate feature importance using specified method (TEST SET ONLY)
    """
    print(f"\nCalculating feature importance using method: {args.importance_method}")
    print(f"Importance scope: {args.importance_scope}")
    print(f"Data source: TEST SET ONLY")
    print(f"Disease-specific importance: {'Yes' if args.disease_specific else 'No'}")
    print(f"Using class weights: {'Yes' if args.use_class_weights else 'No'}")
    
    model.eval()  # Ensure model is in evaluation mode
    
    if args.importance_scope == "disease_wise":
        return calculate_disease_wise_importance(model, dataloader, disease_labels, device, args)
    else:  # overall
        return calculate_overall_importance(model, dataloader, disease_labels, device, args)

def calculate_disease_wise_importance(model, dataloader, disease_labels, device, args):
    # Initialize importance storage for each disease
    total_importance = {disease: None for disease in disease_labels}
    sample_counts = {disease: 0 for disease in disease_labels}
    
    # Calculate class weights if needed
    if args.use_class_weights:
        class_weights = {}
        all_labels = []
        for _, _, labels in dataloader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        
        for disease_idx, disease in enumerate(disease_labels):
            pos_count = all_labels[:, disease_idx].sum().item()
            neg_count = len(all_labels) - pos_count
            total = len(all_labels)
            
            pos_weight = total / (2 * pos_count) if pos_count > 0 else 1.0
            neg_weight = total / (2 * neg_count) if neg_count > 0 else 1.0
            
            class_weights[disease] = {
                'positive': pos_weight,
                'negative': neg_weight
            }
            print(f"  {disease} - Positive weight: {pos_weight:.3f}, Negative weight: {neg_weight:.3f}")
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for batch_idx, (batch_input, covariates, batch_targets) in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        
        batch_input = batch_input.to(device)
        covariates = covariates.to(device)
        batch_targets = batch_targets.to(device)
        
        # Process each disease separately
        for disease_idx, disease in enumerate(disease_labels):
            disease_targets = batch_targets[:, disease_idx]
            
            # Filter samples if disease-specific
            if args.disease_specific:
                positive_mask = disease_targets == 1
                if positive_mask.sum() == 0:
                    continue # Skip if no positive cases in batch
                
                filtered_input = batch_input[positive_mask]
                filtered_covariates = covariates[positive_mask] if covariates.numel() > 0 else None
                filtered_targets = disease_targets[positive_mask]
            else:
                filtered_input = batch_input
                filtered_covariates = covariates
                filtered_targets = disease_targets
            
            batch_size = filtered_input.size(0)
            if batch_size == 0:
                continue
            
            if args.importance_method == "loss_based":
                # Loss-based gradient method for single disease
                filtered_input.requires_grad_(True)
                model.zero_grad()
                
                # Forward pass
                outputs = model(filtered_input, filtered_covariates)
                disease_logits = outputs[:, disease_idx]
                
                # Compute loss
                loss_per_sample = criterion(disease_logits, filtered_targets)
                
                # Apply class weights if enabled
                if args.use_class_weights:
                    weights = torch.where(
                        filtered_targets == 1,
                        torch.tensor(class_weights[disease]['positive']).to(device),
                        torch.tensor(class_weights[disease]['negative']).to(device)
                    )
                    loss_per_sample = loss_per_sample * weights
                
                total_loss = loss_per_sample.mean()

                # Backward pass
                total_loss.backward()
                
                if filtered_input.grad is not None:
                    gradients = filtered_input.grad.detach()
                    # Average across channels
                    batch_importance = gradients.abs().mean(dim=(0, 2)).cpu().numpy()  # [n_snps]
                    
                    if total_importance[disease] is None:
                        total_importance[disease] = np.zeros_like(batch_importance)
                    
                    total_importance[disease] += batch_importance * batch_size
                    sample_counts[disease] += batch_size
                
                filtered_input.grad = None
                
            elif args.importance_method == "integrated_gradients":
                # Integrated gradients method
                baseline = create_baseline(filtered_input, args.baseline_type)
                
                # Calculate integrated gradients
                batch_importance = integrated_gradients(
                    model, filtered_input, filtered_covariates, 
                    disease_idx, baseline, args.ig_steps, device,
                    debug=args.debug_gradients
                )
                
                # Average across batch
                avg_importance = batch_importance.mean(dim=0).cpu().numpy()  # [n_snps]
                
                if total_importance[disease] is None:
                    total_importance[disease] = np.zeros_like(avg_importance)
                
                total_importance[disease] += avg_importance * batch_size
                sample_counts[disease] += batch_size
    
    # Normalize by number of samples processed
    feature_importance = {}
    for disease in disease_labels:
        if total_importance[disease] is not None and sample_counts[disease] > 0:
            feature_importance[disease] = total_importance[disease] / sample_counts[disease]
            print(f"Completed importance calculation for {disease} ({sample_counts[disease]} samples)")
        else:
            print(f"Warning: No samples processed for {disease}")
            first_disease = next(d for d in disease_labels if total_importance[d] is not None)
            feature_importance[disease] = np.zeros_like(total_importance[first_disease])
    
    return feature_importance

def calculate_overall_importance(model, dataloader, disease_labels, device, args):
    print(f"Disease weighting strategy: {args.disease_weights}")
    
    # Calculate disease weights
    disease_weights = get_disease_weights(dataloader, disease_labels, args)
    
    total_importance = None
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for batch_idx, (batch_input, covariates, batch_targets) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Processing first batch of {len(dataloader)} (Overall mode)")

        if (batch_idx + 1) % 10 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)} (Overall mode)")
        
        batch_input = batch_input.to(device)
        covariates = covariates.to(device)
        batch_targets = batch_targets.to(device)
        
        # Apply disease-specific filtering if needed
        if args.disease_specific:
            # In overall mode with disease_specific, use samples that are positive for ANY disease
            any_positive_mask = batch_targets.sum(dim=1) > 0
            if any_positive_mask.sum() == 0:
                continue
            
            filtered_input = batch_input[any_positive_mask]
            filtered_covariates = covariates[any_positive_mask] if covariates.numel() > 0 else None
            filtered_targets = batch_targets[any_positive_mask]
        else:
            filtered_input = batch_input
            filtered_covariates = covariates
            filtered_targets = batch_targets
        
        batch_size = filtered_input.size(0)
        if batch_size == 0:
                continue
        
        if args.importance_method == "loss_based":
            # Multi-task loss approach
            filtered_input.requires_grad_(True)
            model.zero_grad()
            
            outputs = model(filtered_input, filtered_covariates)
            
            # Calculate weighted loss across all diseases
            total_loss = 0
            for disease_idx, disease in enumerate(disease_labels):
                disease_logits = outputs[:, disease_idx]
                disease_targets = filtered_targets[:, disease_idx]
                
                loss_per_sample = criterion(disease_logits, disease_targets)
                disease_loss = loss_per_sample.mean()
                
                # Apply disease weight
                weighted_loss = disease_loss * disease_weights[disease]
                total_loss += weighted_loss
            
            # Single backward pass for all diseases
            total_loss.backward()
            
            if filtered_input.grad is not None:
                gradients = filtered_input.grad.detach()
                batch_importance = gradients.abs().mean(dim=(0, 2)).cpu().numpy()
                
                if total_importance is None:
                    total_importance = np.zeros_like(batch_importance)
                
                total_importance += batch_importance * batch_size
                total_samples += batch_size
            
            filtered_input.grad = None
            
        elif args.importance_method == "integrated_gradients":
            # For IG in overall mode, average importance across diseases
            baseline = create_baseline(filtered_input, args.baseline_type)
            
            batch_importance_sum = None
            
            for disease_idx, disease in enumerate(disease_labels):
                disease_importance = integrated_gradients(
                    model, filtered_input, filtered_covariates, 
                    disease_idx, baseline, args.ig_steps, device,
                    debug=args.debug_gradients
                )
                
                # Weight by disease importance
                weighted_importance = disease_importance * disease_weights[disease]

                print(f"Computed weights for batch {batch_idx + 1}/{len(dataloader)} with shape {weighted_importance.shape} : (Overall mode)")
                
                if batch_importance_sum is None:
                    batch_importance_sum = weighted_importance
                else:
                    batch_importance_sum += weighted_importance
            
            # Average across batch
            avg_importance = batch_importance_sum.mean(dim=0).cpu().numpy()
            
            if total_importance is None:
                total_importance = np.zeros_like(avg_importance)
            
            total_importance += avg_importance * batch_size
            total_samples += batch_size
    
    # Normalize by total samples
    if total_samples > 0:
        overall_importance = total_importance / total_samples
    else:
        print("Warning: No samples processed for overall importance")
        overall_importance = np.zeros(total_importance.shape[0])
    
    print(f"Completed overall importance calculation ({total_samples} samples)")
    
    # Return in format compatible with existing code
    return {"overall": overall_importance}

def get_disease_weights(dataloader, disease_labels, args):
    """Calculate weights for diseases in overall mode"""
    if args.disease_weights == "equal":
        weights = {disease: 1.0 for disease in disease_labels}
        
    elif args.disease_weights == "balanced":
        # Weight inversely proportional to disease prevalence
        all_labels = []
        for _, _, labels in dataloader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        
        weights = {}
        for disease_idx, disease in enumerate(disease_labels):
            pos_count = all_labels[:, disease_idx].sum().item()
            total_count = len(all_labels)
            prevalence = pos_count / total_count if total_count > 0 else 1.0
            
            # Inverse prevalence weighting
            weights[disease] = 1.0 / (prevalence + 1e-6)  # Avoid division by zero
            
    elif args.disease_weights == "custom":
        # Parse custom weights
        custom_values = [float(x.strip()) for x in args.custom_weights.split(',')]
        if len(custom_values) != len(disease_labels):
            raise ValueError(f"Number of custom weights ({len(custom_values)}) must match number of diseases ({len(disease_labels)})")
        weights = dict(zip(disease_labels, custom_values))
    
    # Normalize weights to sum to number of diseases
    total_weight = sum(weights.values())
    normalized_weights = {k: v * len(disease_labels) / total_weight for k, v in weights.items()}
    
    print(f"Disease weights: {normalized_weights}")
    return normalized_weights

def test_set_sample_selection(test_files, phenotype_data, disease_labels, max_samples, 
                             sampling_strategy="stratified", min_positive_per_disease=50, random_seed=42):
    """
    Sample selection specifically for test set data
    """
    
    np.random.seed(random_seed)
    
    print(f"\nTest Set Sample Selection Strategy: {sampling_strategy}")
    print(f"Available test samples: {len(test_files)}")
    print(f"Requested max samples: {max_samples}")
    
    if len(test_files) <= max_samples:
        print(f"Using all {len(test_files)} test samples (less than max_samples)")
        return test_files
    
    # Extract sample IDs from file paths
    sample_ids = []
    for file_path in test_files:
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        sample_ids.append(sample_id)
    
    # Create a mapping from sample_id to file_path
    sample_to_file = dict(zip(sample_ids, test_files))
    
    # Get phenotype data for available test samples
    available_phenotype = phenotype_data[phenotype_data['new_order'].isin(sample_ids)].copy()
    
    if sampling_strategy == "first":
        selected_files = test_files[:max_samples]
        print(f"Selected first {len(selected_files)} test samples")
        
    elif sampling_strategy == "random":
        selected_indices = np.random.choice(len(test_files), max_samples, replace=False)
        selected_files = [test_files[i] for i in selected_indices]
        print(f"Randomly selected {len(selected_files)} test samples")
        
    elif sampling_strategy == "stratified":
        selected_sample_ids = set()
        
        # First, ensure we have enough positive cases for each disease
        for disease in disease_labels:
            disease_positive = available_phenotype[available_phenotype[disease] == 1]
            n_positive = len(disease_positive)
            
            print(f"Disease {disease}: {n_positive} positive cases in test set")
            
            if n_positive > 0:
                n_take = min(n_positive, min_positive_per_disease)
                selected_positive = disease_positive.sample(n=n_take, random_state=random_seed)
                selected_sample_ids.update(selected_positive['new_order'].tolist())
        
        # Fill remaining slots with random sampling
        remaining_slots = max_samples - len(selected_sample_ids)
        if remaining_slots > 0:
            remaining_samples = available_phenotype[
                ~available_phenotype['new_order'].isin(selected_sample_ids)
            ]
            
            if len(remaining_samples) > 0:
                n_additional = min(len(remaining_samples), remaining_slots)
                additional_samples = remaining_samples.sample(n=n_additional, random_state=random_seed)
                selected_sample_ids.update(additional_samples['new_order'].tolist())
        
        selected_files = [sample_to_file[sid] for sid in selected_sample_ids if sid in sample_to_file]
        print(f"Stratified sampling selected {len(selected_files)} test samples")
        
    elif sampling_strategy == "balanced":
        selected_sample_ids = set()
        
        for disease in disease_labels:
            disease_positive = available_phenotype[available_phenotype[disease] == 1]
            disease_negative = available_phenotype[available_phenotype[disease] == 0]
            
            n_positive = len(disease_positive)
            n_negative = len(disease_negative)
            
            print(f"Disease {disease}: {n_positive} positive, {n_negative} negative in test set")
            
            n_take_each = min(min_positive_per_disease, n_positive, n_negative)
            
            if n_take_each > 0:
                selected_pos = disease_positive.sample(n=n_take_each, random_state=random_seed)
                selected_neg = disease_negative.sample(n=n_take_each, random_state=random_seed)
                
                selected_sample_ids.update(selected_pos['new_order'].tolist())
                selected_sample_ids.update(selected_neg['new_order'].tolist())
        
        # Fill remaining slots if needed
        remaining_slots = max_samples - len(selected_sample_ids)
        if remaining_slots > 0:
            remaining_samples = available_phenotype[
                ~available_phenotype['new_order'].isin(selected_sample_ids)
            ]
            if len(remaining_samples) > 0:
                n_additional = min(len(remaining_samples), remaining_slots)
                additional_samples = remaining_samples.sample(n=n_additional, random_state=random_seed)
                selected_sample_ids.update(additional_samples['new_order'].tolist())
        
        selected_files = [sample_to_file[sid] for sid in selected_sample_ids if sid in sample_to_file]
        print(f"Balanced sampling selected {len(selected_files)} test samples")
    
    # Print final disease distribution in test set
    final_sample_ids = [int(f.split('sample_')[1].split('.npy')[0]) for f in selected_files]
    final_phenotype = available_phenotype[available_phenotype['new_order'].isin(final_sample_ids)]
    
    print(f"\nFinal test sample composition:")
    for disease in disease_labels:
        n_pos = (final_phenotype[disease] == 1).sum()
        n_neg = (final_phenotype[disease] == 0).sum()
        total = len(final_phenotype)
        pos_pct = (n_pos / total * 100) if total > 0 else 0
        print(f"  {disease}: {n_pos} positive ({pos_pct:.1f}%), {n_neg} negative")
    
    return selected_files

def load_normalizers_from_checkpoint(checkpoint_path):
    """
    Load normalizers from the training checkpoint
    """
    print("Loading normalizers from checkpoint...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check if normalizers are stored in checkpoint
        if 'normalizers' in checkpoint:
            normalizers = checkpoint['normalizers']
            print("✓ Loaded normalizers from checkpoint")
            return normalizers
        else:
            print("⚠ No normalizers found in checkpoint - will create default ones")
            return None
            
    except Exception as e:
        print(f"✗ Error loading normalizers from checkpoint: {e}")
        return None

def create_default_normalizers(phenotype_data, args):
    """
    Create and fit normalizers on all available data (fallback if not in checkpoint)
    Note: This is less ideal than using training-fitted normalizers
    """
    print("⚠ Creating default normalizers (not ideal - should use training-fitted ones)")
    
    normalizers = {
        'age': CovariateNormalizer(args.norm_age if hasattr(args, 'norm_age') else "none"),
        'pcs': CovariateNormalizer(args.norm_pcs if hasattr(args, 'norm_pcs') else "none"),
        'gender': CovariateNormalizer(args.norm_gender if hasattr(args, 'norm_gender') else "none"),
        'bmi': CovariateNormalizer(args.norm_bmi if hasattr(args, 'norm_bmi') else "none")
    }

    # Fit normalizers on available data (if columns exist)
    if 'Agexit' in phenotype_data.columns:
        all_age = phenotype_data['Agexit'].values
        normalizers['age'].fit(all_age)
    if any([f'PC{i}' in phenotype_data.columns for i in range(1, args.num_covariates + 1)]):
        all_pcs = np.array([phenotype_data[f'PC{i}'].values for i in range(1, args.num_covariates + 1)]).T
        normalizers['pcs'].fit(all_pcs)
    if 'Sex' in phenotype_data.columns:
        all_gender = phenotype_data['Sex'].values
        normalizers['gender'].fit(all_gender)
    if 'Bmi_C' in phenotype_data.columns:
        all_bmi = phenotype_data['Bmi_C'].values
        normalizers['bmi'].fit(all_bmi)
    
    return normalizers


def load_model_and_data(args):
    """Load the trained model and prepare TEST SET data only"""
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model state dict
    if 'best_model_state_dict' in checkpoint and checkpoint['best_model_state_dict'] is not None:
        model_state_dict = checkpoint['best_model_state_dict']
        print("Using best model weights")
    else:
        model_state_dict = checkpoint['model_state_dict']
        if model_state_dict is None:
            raise KeyError("No model_state_dict found in checkpoint")
        print("Using current model weights")
    
    # Load phenotype data
    print("Loading phenotype data...")
    if not os.path.exists(args.phenotype_file):
        raise FileNotFoundError(f"Phenotype file not found: {args.phenotype_file}")
    phenotype_data = pd.read_excel(args.phenotype_file)
    
    # Parse disease labels
    disease_labels = [label.strip() for label in args.disease_labels.split(',')]
    
    # SAME FILTERING LOGIC AS TRAINING SCRIPT
    print("REPRODUCING EXACT TRAINING DATA FILTERING")
    print("="*60)
    
    # Get list of genotype files and sort numerically by sample number
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.npy"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.npy')[0]))
    print(f"Number of genotype files found: {len(file_list)}")

    # Get the unique sample IDs from the phenotype data
    phenotype_samples = set(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {len(phenotype_samples)}")

    # Filter the file list to only include files for samples in the phenotype data
    filtered_file_list = []
    for file_path in file_list:
        # Extract sample ID from filename
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        
        # Check if this sample ID is in the phenotype data
        if sample_id in phenotype_samples:
            filtered_file_list.append(file_path)

    print(f"Number of filtered genotype files matching phenotype data: {len(filtered_file_list)}")

    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file after filtering is: {first_genotype_file}")
        input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined input size: {input_size}")
    else:
        print("No matching genotype files found!")
        raise ValueError("No genotype files found matching phenotype data")

    if len(filtered_file_list) != len(phenotype_samples):
        print(f"Warning: Number of files ({len(filtered_file_list)}) does not match number of samples ({len(phenotype_samples)}) in phenotype data.")
        
        # Additional diagnostics
        file_sample_ids = set(int(f.split('sample_')[1].split('.npy')[0]) for f in filtered_file_list)
        missing_in_files = phenotype_samples - file_sample_ids
        missing_in_phenotype = file_sample_ids - phenotype_samples
        
        if missing_in_files:
            print(f"  - {len(missing_in_files)} samples in phenotype data but missing genotype files")
            if len(missing_in_files) <= 10:
                print(f"    Missing file samples: {sorted(list(missing_in_files))}")
        
        if missing_in_phenotype:
            print(f"  - {len(missing_in_phenotype)} genotype files but missing phenotype data")
            if len(missing_in_phenotype) <= 10:
                print(f"    Missing phenotype samples: {sorted(list(missing_in_phenotype))}")
    
    print(f"✓ Using {len(filtered_file_list)} samples that have both genotype and phenotype data")
    
    # RECREATE EXACT TRAIN/TEST SPLIT
    print("\n" + "="*60)
    print("RECREATING EXACT TRAINING DATA SPLIT")
    print("="*60)
    
    # Split data using EXACTLY the same parameters as training
    train_files, test_files = train_test_split(filtered_file_list, test_size=args.test_size, random_state=args.random_state)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")
    
    # Verify this matches what would be expected
    expected_test_size = int(len(filtered_file_list) * args.test_size)
    actual_test_size = len(test_files)
    print(f"Expected test set size: ~{expected_test_size}, Actual: {actual_test_size}")
    
    print("\n" + "="*50)
    print("USING TEST SET ONLY FOR FEATURE IMPORTANCE")
    print("="*50)
    print(f"Using {len(test_files)} test set samples for analysis")
    
    # Sample selection from test set
    if len(test_files) > args.max_samples:
        selected_files = test_set_sample_selection(
            test_files, 
            phenotype_data, 
            disease_labels, 
            args.max_samples,
            sampling_strategy=args.sampling_strategy,
            min_positive_per_disease=args.min_positive_per_disease
        )
        print(f"Selected {len(selected_files)} test samples using {args.sampling_strategy} strategy")
    else:
        selected_files = test_files
        print(f"Using all {len(selected_files)} available test samples")
    
    # Load normalizers from checkpoint
    normalizers = load_normalizers_from_checkpoint(args.checkpoint_path)
    
    if normalizers is None:
        # Fallback: create normalizers (less ideal)
        normalizers = create_default_normalizers(phenotype_data, args)
    
    print(f"\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    print(f"  - Input size: {input_size:,} SNPs")
    print(f"  - Diseases: {len(disease_labels)}")
    print(f"  - Use covariates: {bool(args.cov)}")
    print(f"  - Use age: {bool(args.use_age)}")
    print(f"  - Use gender: {bool(args.use_gender)}")
    print(f"  - hardcoded_kernels: {bool(args.hardcoded_kernels)}")
    print(f"  - hardcoded_strides: {bool(args.hardcoded_strides)}")
    
    model = MultilabelGenotypeModel(
        input_size=input_size,
        num_diseases=len(disease_labels),
        kernel_sizes=args.kernel_sizes,
        stride=args.stride,
        conv_channels=args.conv_channels,
        fc_layers=args.fc_layers,
        act=args.act,
        dropout_rate=args.dropout,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        num_covariates=args.num_covariates,
        use_pooling=bool(args.use_pooling),
        pool_size=args.pool_size,
        pool_type=args.pool_type,
        use_multi_scale=bool(args.use_multi_scale),
        use_disease_attention=bool(args.use_disease_attention),
        use_separate_heads=bool(args.use_separate_heads),
        attention_heads=args.attention_heads,
        attention_dim=args.attention_dim,
        multi_scale_kernels=args.multi_scale_kernels,
        multi_scale_strides=args.multi_scale_strides,
        multi_scale_fusion=args.multi_scale_fusion,
        multi_scale_mode=args.multi_scale_mode,
        hardcoded_kernels=args.hardcoded_kernels,
        hardcoded_strides=args.hardcoded_strides,
        use_pointwise_conv=bool(args.use_pointwise_conv),
        pointwise_channels=args.pointwise_channels,
        use_transformer=bool(args.use_transformer),
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dim=args.transformer_dim,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        use_positional_encoding=bool(args.use_positional_encoding),
        max_seq_len=args.max_seq_len,
        use_cls_token=bool(args.use_cls_token),
        use_covariate_tokens=bool(args.use_covariate_tokens),
        covariate_embed_dim=args.covariate_embed_dim,
        covariate_token_strategy=args.covariate_token_strategy,
        pooling_strategy=args.pooling_strategy,
        pooling_kwargs=None
    )
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create dataset with proper normalizers
    dataset = SimpleGenotypeDataset(
        selected_files, 
        phenotype_data, 
        disease_labels, 
        normalizers=normalizers,
        use_covariates=bool(args.cov), 
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender), 
        use_bmi=bool(args.use_bmi),
        num_covariates=args.num_covariates
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Model and data loaded successfully")

    return model, dataloader, disease_labels, input_size

def save_results(feature_importance, disease_labels, output_dir, args):
    """Save results for both disease-wise and overall modes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create method suffix for filenames
    method_suffix = f"{args.importance_method}_{args.importance_scope}_test_set"
    if args.disease_specific:
        method_suffix += "_disease_specific"
    if args.use_class_weights:
        method_suffix += "_weighted"
    
    results = {}
    
    if args.importance_scope == "disease_wise":
        # Original disease-wise saving
        for disease in disease_labels:
            importance = feature_importance[disease]

            num_snps = importance.shape[0]

            df = pd.DataFrame({
                'SNP_Index': np.arange(num_snps),    #  0-based SNP index
                'Importance_Score': importance
            })

            # Optional: sort by importance descending
            df = df.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
            df["Rank"] = np.arange(1, num_snps + 1)  #  1-based ranking

            # Save results
            csv_path = os.path.join(output_dir, f'{disease}_all_snps_importance_{method_suffix}.csv')
            df.to_csv(csv_path, index=False)
            
            scores_path = os.path.join(output_dir, f'{disease}_all_snps_importance_scores_{method_suffix}.npy')
            np.save(scores_path, importance)
            
            results[disease] = df
            
            print(f"\n{disease.upper()} Results:")
            print(f"  - Method: {method_suffix}")
            print(f"  - Files saved: {csv_path}")
    
    else:  # overall mode
        importance = feature_importance["overall"]
        num_snps = importance.shape[0]
        df = pd.DataFrame({
            'SNP_Index': np.arange(num_snps),
            'Importance_Score': importance
        })

        df = df.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
        df["Rank"] = np.arange(1, num_snps + 1)
        
        csv_path = os.path.join(output_dir, f'overall_all_snps_importance_{method_suffix}.csv')
        df.to_csv(csv_path, index=False)
        
        scores_path = os.path.join(output_dir, f'overall_all_snps_importance_scores_{method_suffix}.npy')
        np.save(scores_path, importance)
        
        results["overall"] = df
        
        print(f"\nOVERALL Results:")
        print(f"  - Disease weighting: {args.disease_weights}")
        print(f"  - Method: {method_suffix}")
        print(f"  - Files saved: {csv_path}")
    
    # Save method configuration
    config_path = os.path.join(output_dir, f'importance_config_{method_suffix}.txt')
    with open(config_path, 'w') as f:
        f.write(f"Data source: TEST SET ONLY\n")
        f.write(f"Importance scope: {args.importance_scope}\n")
        f.write(f"Method: {args.importance_method}\n")
        f.write(f"Disease-specific: {args.disease_specific}\n")
        f.write(f"Use class weights: {args.use_class_weights}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Random state: {args.random_state}\n")
        f.write(f"Sampling strategy: {args.sampling_strategy}\n")
        f.write(f"Max samples: {args.max_samples}\n")
        if args.importance_scope == "overall":
            f.write(f"Disease weights: {args.disease_weights}\n")
            if args.disease_weights == "custom":
                f.write(f"Custom weights: {args.custom_weights}\n")
        if "integrated_gradients" in args.importance_method:
            f.write(f"IG steps: {args.ig_steps}\n")
            f.write(f"Baseline type: {args.baseline_type}\n")
    
    print(f"\nConfiguration saved to: {config_path}")

    return results

def main():
    args = parse_args()
    
    print("SNP FEATURE IMPORTANCE SCORE COMPUTATION -  TEST SET ONLY ANALYSIS")
    print("=" * 80)
    
    # # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"\nUsing device: {device}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Aborting.")

    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Parse disease labels
    disease_labels = [label.strip() for label in args.disease_labels.split(',')]
    
    # Verify split parameters match training
    print(f"\nData Split Parameters:")
    print(f"  - Test size ratio: {args.test_size}")
    print(f"  - Random state: {args.random_state}")
    print(f"  ⚠ ENSURE these match your training script parameters!")
    
    # Validate hardcoded parameters if using hardcoded mode
    if args.use_multi_scale and args.multi_scale_mode == "hardcoded":
        validate_hardcoded_parameters(args)


    # Load model and data (test set only)
    model, dataloader, disease_labels, input_size = load_model_and_data(args)
    model = model.to(device)

    print(f"Model loaded successfully:")
    print(f"  - Input size: {input_size:,} SNPs")
    print(f"  - Diseases: {', '.join(disease_labels)}")
    print(f"  - Test samples for analysis: {len(dataloader.dataset)}")
    print(f"  - Sampling strategy: {args.sampling_strategy}")
    
    feature_importance = calculate_feature_importance(model, dataloader, disease_labels, device, args)
    
     # Save results
    results = save_results(feature_importance, disease_labels, args.output_dir, args)
    
    print(f"\n" + "=" * 80)
    print("EXTRACTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Analysis mode: {args.importance_scope}")
    print(f"Data source: TEST SET ONLY")
    print(f"Method: {args.importance_method}")


if __name__ == '__main__':
    main()