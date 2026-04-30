import os
import torch
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Extract top SNPs based on model weights")
    parser.add_argument("-checkpoint_path", type=str, default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/01_snps_5d_ks/checkpoint_epoch_70.pt", help="Path to the trained model checkpoint")
    parser.add_argument("-output_dir", type=str, default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/01_snps_5d_ks/", help="Directory to save the output files")
    parser.add_argument("-top_k", type=int, default=90000, help="Number of top SNPs to extract (default: 90000)")
    parser.add_argument("-input_size", type=int, default=5111472, help="Total number of SNPs in the input. If None, will be estimated from model")
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """Load the model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model state dict - try different possible keys
    if 'best_model_state_dict' in checkpoint and checkpoint['best_model_state_dict'] is not None:
        model_state_dict = checkpoint['best_model_state_dict']
        print("Using best model weights from checkpoint")
    elif 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print("Using current model weights from checkpoint")
    else:
        # Checkpoint might be just the state dict
        model_state_dict = checkpoint
        print("Using checkpoint as state dict directly")
    
    return model_state_dict, checkpoint

def extract_conv_weights(model_state_dict):
    """Extract weights from the first convolutional layer"""
    
    # Find the first convolutional layer weights
    conv1_weight_key = None
    for key in model_state_dict.keys():
        if 'conv_layers.0.weight' in key or (key.startswith('conv_layers') and 'weight' in key):
            conv1_weight_key = key
            break
    
    if conv1_weight_key is None:
        print("Available keys in model state dict:")
        for key in model_state_dict.keys():
            print(f"  {key}")
        raise ValueError("Could not find first convolutional layer weights")
    
    print(f"Found first conv layer weights at key: {conv1_weight_key}")
    conv1_weights = model_state_dict[conv1_weight_key]
    
    print(f"Conv1 weights shape: {conv1_weights.shape}")
    # Expected shape: [out_channels, in_channels, kernel_size]
    
    return conv1_weights

def calculate_snp_importance_from_conv_weights(conv_weights, input_size, stride=64):
    """
    Calculate SNP importance scores from convolutional weights
    
    Args:
        conv_weights: Tensor of shape [out_channels, in_channels, kernel_size]
        input_size: Total number of SNPs in the input
        stride: Stride of the convolutional layer
    
    Returns:
        snp_importance: Array of importance scores for each SNP position
    """
    
    out_channels, in_channels, kernel_size = conv_weights.shape
    print(f"Calculating importance for {input_size} SNPs using conv weights:")
    print(f"  - Output channels: {out_channels}")
    print(f"  - Input channels: {in_channels}")  
    print(f"  - Kernel size: {kernel_size}")
    print(f"  - Stride: {stride}")
    
    # Initialize importance array
    snp_importance = np.zeros(input_size)
    snp_count = np.zeros(input_size)  # Track how many times each SNP is counted
    
    # Convert to numpy for easier manipulation
    weights_np = conv_weights.detach().cpu().numpy()
    
    # Calculate the number of positions the conv layer will produce
    conv_output_size = (input_size - kernel_size) // stride + 1
    print(f"  - Conv output size: {conv_output_size}")
    
    # For each position in the convolutional output
    for pos in range(conv_output_size):
        # Calculate which SNPs this position looks at
        start_snp = pos * stride
        end_snp = start_snp + kernel_size
        
        # Make sure we don't go beyond input size
        end_snp = min(end_snp, input_size)
        actual_kernel_size = end_snp - start_snp
        
        if actual_kernel_size <= 0:
            break
            
        # Get the weights for this position (sum across all filters and input channels)
        # Shape: [out_channels, in_channels, actual_kernel_size]
        position_weights = weights_np[:, :, :actual_kernel_size]
        
        # Sum absolute weights across output channels and input channels
        # This gives us the total "influence" of each SNP position in this kernel
        position_importance = np.sum(np.abs(position_weights), axis=(0, 1))
        
        # Add to the corresponding SNP positions
        snp_importance[start_snp:end_snp] += position_importance
        snp_count[start_snp:end_snp] += 1
    
    # Average importance by the number of times each SNP was included
    # (to account for overlapping kernels)
    mask = snp_count > 0
    snp_importance[mask] = snp_importance[mask] / snp_count[mask]
    
    print(f"SNP importance calculation completed")
    print(f"  - Min importance: {np.min(snp_importance):.6f}")
    print(f"  - Max importance: {np.max(snp_importance):.6f}")
    print(f"  - Mean importance: {np.mean(snp_importance):.6f}")
    print(f"  - SNPs with non-zero importance: {np.sum(snp_importance > 0)}")
    
    return snp_importance

def estimate_input_size_from_model(model_state_dict, stride=64, pool_size=64):
    """
    Estimate the input size based on the model architecture
    This is a rough estimation - you might need to adjust based on your specific model
    """
    
    # Try to find fully connected layer to work backwards
    fc_keys = [k for k in model_state_dict.keys() if 'fc_shared' in k and 'weight' in k and '0.' in k]
    
    if fc_keys:
        fc_weight = model_state_dict[fc_keys[0]]
        fc_input_size = fc_weight.shape[1]
        print(f"First FC layer input size: {fc_input_size}")
        
        # Account for covariates (typically 10 PCs + age + gender = 12)
        covariates_size = 12
        conv_output_size = fc_input_size - covariates_size
        
        # Work backwards from conv output size
        # conv_output_size = pool_size * out_channels
        conv_channels = None
        for key in model_state_dict.keys():
            if 'conv_layers' in key and 'weight' in key:
                # Get the last conv layer
                conv_channels = model_state_dict[key].shape[0]
        
        if conv_channels:
            # After pooling: pool_size * conv_channels
            expected_after_pooling = pool_size * conv_channels
            print(f"Expected size after pooling: {expected_after_pooling}")
            print(f"Actual conv output size: {conv_output_size}")
            
        # This is a rough estimate - you may need to adjust
        estimated_input = 5000000  # 5M SNPs as mentioned in the script
        print(f"Using estimated input size: {estimated_input}")
        return estimated_input
    
    return None

def save_results(snp_importance, top_k, output_dir):
    """Save the results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top K SNP indices
    top_indices = np.argsort(snp_importance)[::-1][:top_k]
    top_scores = snp_importance[top_indices]
    
    print(f"\nTop {top_k} SNPs:")
    print(f"  - Highest importance score: {top_scores[0]:.6f}")
    print(f"  - Lowest importance score: {top_scores[-1]:.6f}")
    print(f"  - Score range: {top_scores[0] - top_scores[-1]:.6f}")
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'SNP_Index': top_indices,
        'Importance_Score': top_scores,
        'Rank': range(1, top_k + 1)
    })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'top_{top_k}_snps.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save just the indices (for easy loading)
    indices_path = os.path.join(output_dir, f'top_{top_k}_snp_indices.txt')
    np.savetxt(indices_path, top_indices, fmt='%d')
    print(f"SNP indices saved to: {indices_path}")
    
    # Save all importance scores
    all_scores_path = os.path.join(output_dir, 'all_snp_importance_scores.npy')
    np.save(all_scores_path, snp_importance)
    print(f"All importance scores saved to: {all_scores_path}")
    
    return results_df

def main():
    args = parse_args()
    
    print("="*60)
    print("SNP IMPORTANCE EXTRACTION FROM TRAINED MODEL")
    print("="*60)
    
    # Load checkpoint
    model_state_dict, checkpoint = load_checkpoint(args.checkpoint_path)
    
    # Extract convolutional weights
    conv_weights = extract_conv_weights(model_state_dict)
    
    # Determine input size
    if args.input_size is None:
        input_size = estimate_input_size_from_model(model_state_dict)
        if input_size is None:
            print("Could not estimate input size. Please provide it manually using -input_size argument")
            return
    else:
        input_size = args.input_size
    
    print(f"\nUsing input size: {input_size:,} SNPs")
    
    # Extract model hyperparameters from checkpoint if available
    stride = 2  # Default value
    if 'hyperparameters' in checkpoint:
        # Try to extract stride from hyperparameters if saved
        pass
    
    # Calculate SNP importance
    snp_importance = calculate_snp_importance_from_conv_weights(
        conv_weights, input_size, stride=stride
    )
    
    # Save results
    results_df = save_results(snp_importance, args.top_k, args.output_dir)
    
    print(f"\n" + "="*60)
    print("EXTRACTION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Top {args.top_k} SNPs extracted and saved to {args.output_dir}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  - Total SNPs analyzed: {len(snp_importance):,}")
    print(f"  - Top SNPs extracted: {args.top_k:,}")
    print(f"  - Percentage of SNPs selected: {(args.top_k/len(snp_importance)*100):.2f}%")
    
    # Show top 10 SNPs as example
    print(f"\nTop 10 SNPs (as example):")
    print(results_df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()