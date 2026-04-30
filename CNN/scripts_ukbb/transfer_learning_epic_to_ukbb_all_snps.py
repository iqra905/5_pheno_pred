import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict

class EPICtoUKBBWeightAdapter:
    """
    Adapts EPIC-trained model weights to UKBB architecture.
    Handles SNP remapping and layer-wise weight transfer.
    """
    
    def __init__(self, epic_checkpoint_path, common_snps_mapping=None):
        """
        Args:
            epic_checkpoint_path: Path to EPIC model checkpoint
            common_snps_mapping: Dict or file with SNP index mappings
                Format: {'epic_idx': 'ukbb_idx'} or {'epic_idx': {'ukbb_idx': ..., 'snp_id': ...}}
        """
        self.epic_checkpoint = torch.load(epic_checkpoint_path, map_location='cpu', weights_only=False)
        self.common_snps_mapping = common_snps_mapping
        self.epic_model_state = self.epic_checkpoint.get('model_state_dict', self.epic_checkpoint)
        
        print(f"✓ Loaded EPIC checkpoint from: {epic_checkpoint_path}")
        print(f"✓ EPIC model keys: {list(self.epic_model_state.keys())[:5]}...")
    
    def load_snp_mapping(self, mapping_path):
        """Load SNP mappings from file (JSON, NPY, or PKL format)"""
        if mapping_path.endswith('.json'):
            import json
            with open(mapping_path, 'r') as f:
                self.common_snps_mapping = json.load(f)
        elif mapping_path.endswith('.npy'):
            self.common_snps_mapping = np.load(mapping_path, allow_pickle=True).item()
        elif mapping_path.endswith('.pkl'):
            import pickle
            with open(mapping_path, 'rb') as f:
                self.common_snps_mapping = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {mapping_path}")
        
        print(f"✓ Loaded SNP mapping with {len(self.common_snps_mapping)} common SNPs")
        return self.common_snps_mapping
    
    def adapt_conv_weights(self, epic_weight, epic_indices, ukbb_indices):
        """
        Adapt convolution weights from EPIC to UKBB using SNP index mapping.
        
        Args:
            epic_weight: Conv1d weight tensor [out_channels, in_channels, kernel_size]
            epic_indices: List of SNP indices in EPIC dataset
            ukbb_indices: List of SNP indices in UKBB dataset
        
        Returns:
            Adapted weight tensor for UKBB
        """
        out_channels, in_channels, kernel_size = epic_weight.shape
        
        # Create mapping: EPIC SNP position -> UKBB SNP position
        epic_to_ukbb_pos = {}
        for ukbb_pos, ukbb_idx in enumerate(ukbb_indices):
            if ukbb_idx in epic_indices:
                epic_pos = epic_indices.index(ukbb_idx)
                epic_to_ukbb_pos[epic_pos] = ukbb_pos
        
        # Initialize UKBB weights (use small random values to preserve signal)
        ukbb_weight = torch.randn_like(epic_weight) * 0.01
        
        # Copy weights for common SNPs
        copied_count = 0
        for epic_pos, ukbb_pos in epic_to_ukbb_pos.items():
            ukbb_weight[:, :, ukbb_pos] = epic_weight[:, :, epic_pos]
            copied_count += 1
        
        print(f"  Copied {copied_count}/{len(ukbb_indices)} SNP positions")
        return ukbb_weight
    
    def adapt_first_layer(self, ukbb_model, epic_num_snps, ukbb_num_snps):
        """
        Adapt first convolutional layer accounting for different number of SNPs.
        This is the critical layer that takes genomic input.
        
        Strategy:
        1. Use common SNPs mapping to align weights
        2. Interpolate or average weights for non-common SNPs
        """
        print("\n" + "="*80)
        print("ADAPTING FIRST CONVOLUTIONAL LAYER (Genomic Input)")
        print("="*80)
        print(f"EPIC SNPs: {epic_num_snps}, UKBB SNPs: {ukbb_num_snps}")
        
        first_layer_key = 'conv_layers.0.branches'  # First ParallelMultiScaleConvBlock
        
        for branch_idx in range(3):  # 3 branches in parallel
            branch_key = f'{first_layer_key}.{branch_idx}.0'  # Conv1d layer
            
            if branch_key not in self.epic_model_state:
                print(f"Warning: {branch_key} not found in EPIC model")
                continue
            
            epic_weight = self.epic_model_state[f'{branch_key}.weight']  # [out_channels, in_channels, kernel_size]
            epic_bias = self.epic_model_state.get(f'{branch_key}.bias')
            
            out_channels, in_channels, kernel_size = epic_weight.shape
            
            # Strategy 1: Use common SNP mapping if available
            if self.common_snps_mapping and isinstance(self.common_snps_mapping, dict):
                epic_indices = sorted(self.common_snps_mapping.keys())
                ukbb_indices = sorted(self.common_snps_mapping.values()) \
                    if isinstance(list(self.common_snps_mapping.values())[0], int) \
                    else sorted([v['ukbb_idx'] for v in self.common_snps_mapping.values()])
                
                adapted_weight = self.adapt_conv_weights(epic_weight, epic_indices, ukbb_indices)
            else:
                # Strategy 2: If no mapping, use interpolation along SNP dimension
                print(f"  No SNP mapping provided. Using interpolation for {branch_idx}")
                adapted_weight = self._interpolate_weights(epic_weight, epic_num_snps, ukbb_num_snps)
            
            # Update UKBB model with adapted weights
            ukbb_layer_key = f'conv_layers.0.branches.{branch_idx}.0'
            ukbb_model.state_dict()[f'{ukbb_layer_key}.weight'][:] = adapted_weight
            if epic_bias is not None:
                ukbb_model.state_dict()[f'{ukbb_layer_key}.bias'][:] = epic_bias
            
            print(f"✓ Branch {branch_idx}: Adapted weights shape {epic_weight.shape} -> {adapted_weight.shape}")
        
        return ukbb_model
    
    def _interpolate_weights(self, epic_weight, epic_snps, ukbb_snps):
        """
        Interpolate weights along SNP dimension when direct mapping unavailable.
        
        This assumes genomic features are position-dependent and smoother across nearby SNPs.
        """
        out_channels, in_channels, kernel_size = epic_weight.shape
        
        # Create position arrays
        epic_positions = np.linspace(0, 1, epic_snps)
        ukbb_positions = np.linspace(0, 1, ukbb_snps)
        
        # Interpolate each channel separately
        ukbb_weight = torch.zeros(out_channels, in_channels, ukbb_snps)
        
        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                for ks in range(kernel_size):
                    # 1D interpolation along SNP dimension
                    epic_vals = epic_weight[out_ch, in_ch, ks, :].cpu().numpy() if epic_weight.dim() > 3 else epic_weight[out_ch, in_ch, ks].cpu().numpy()
                    
                    # Simple linear interpolation
                    ukbb_vals = np.interp(ukbb_positions, epic_positions, epic_vals)
                    
                    ukbb_weight[out_ch, in_ch, :] = torch.tensor(ukbb_vals, dtype=epic_weight.dtype)
        
        return ukbb_weight
    
    def adapt_remaining_layers(self, ukbb_model):
        """
        Transfer weights from EPIC to UKBB for layers after first conv layer.
        These layers are dimension-dependent and require careful handling.
        """
        print("\n" + "="*80)
        print("ADAPTING REMAINING CONVOLUTIONAL LAYERS")
        print("="*80)
        
        layer_mappings = [
            # Layer 2 (channel expansion 32 -> 64)
            ('conv_layers.1.branches', [
                ('0.0', '0.0'),  # Conv1d weights
                ('0.1', '0.1'),  # BatchNorm
                ('1.0', '1.0'),
                ('1.1', '1.1'),
                ('2.0', '2.0'),
                ('2.1', '2.1'),
            ]),
            # Layer 3 (channel expansion 64 -> 128)
            ('conv_layers.2.branches', [
                ('0.0', '0.0'),
                ('0.1', '0.1'),
                ('1.0', '1.0'),
                ('1.1', '1.1'),
                ('2.0', '2.0'),
                ('2.1', '2.1'),
            ]),
        ]
        
        adapted_layers = 0
        for layer_prefix, mappings in layer_mappings:
            for branch_idx in range(3):
                for epic_suffix, ukbb_suffix in mappings:
                    epic_key = f'{layer_prefix}.{branch_idx}.{epic_suffix}'
                    ukbb_key = f'{layer_prefix}.{branch_idx}.{ukbb_suffix}'
                    
                    if epic_key not in self.epic_model_state:
                        continue
                    
                    try:
                        # Handle Conv1d weights (channel dimensions must match)
                        if 'weight' in epic_key and epic_key.endswith('.weight'):
                            epic_weight = self.epic_model_state[epic_key]
                            ukbb_weight_shape = ukbb_model.state_dict()[ukbb_key].shape
                            
                            if epic_weight.shape == ukbb_weight_shape:
                                ukbb_model.state_dict()[ukbb_key][:] = epic_weight
                                adapted_layers += 1
                            else:
                                print(f"  Shape mismatch {epic_key}: {epic_weight.shape} vs {ukbb_weight_shape}")
                        
                        # Handle biases
                        elif 'bias' in epic_key:
                            epic_bias = self.epic_model_state[epic_key]
                            if epic_bias.shape == ukbb_model.state_dict()[ukbb_key].shape:
                                ukbb_model.state_dict()[ukbb_key][:] = epic_bias
                                adapted_layers += 1
                        
                        # Handle BatchNorm
                        elif any(x in epic_key for x in ['running_mean', 'running_var', 'weight', 'bias']):
                            ukbb_model.state_dict()[ukbb_key][:] = self.epic_model_state[epic_key]
                            adapted_layers += 1
                    
                    except Exception as e:
                        print(f"  Warning: Could not adapt {epic_key}: {str(e)}")
        
        print(f"✓ Successfully adapted {adapted_layers} layer parameters")
        return ukbb_model
    
    def adapt_mamba_layers(self, ukbb_model):
        """
        Transfer Mamba layer weights (dimension-agnostic after flattening).
        Mamba processes the flattened conv output, so direct transfer should work.
        """
        print("\n" + "="*80)
        print("ADAPTING MAMBA LAYERS")
        print("="*80)
        
        mamba_keys = [k for k in self.epic_model_state.keys() if 'mamba' in k]
        
        if not mamba_keys:
            print("No Mamba layers found in EPIC model")
            return ukbb_model
        
        adapted_count = 0
        for epic_key in mamba_keys:
            # Reconstruct UKBB key (should be identical)
            if epic_key in ukbb_model.state_dict():
                try:
                    epic_param = self.epic_model_state[epic_key]
                    ukbb_shape = ukbb_model.state_dict()[epic_key].shape
                    
                    if epic_param.shape == ukbb_shape:
                        ukbb_model.state_dict()[epic_key][:] = epic_param
                        adapted_count += 1
                    else:
                        print(f"  Shape mismatch for {epic_key}: {epic_param.shape} vs {ukbb_shape}")
                except Exception as e:
                    print(f"  Warning: Could not adapt {epic_key}: {str(e)}")
        
        print(f"✓ Adapted {adapted_count} Mamba layer parameters")
        return ukbb_model
    
    def adapt_fc_layers(self, ukbb_model, transfer_mode='partial'):
        """
        Transfer fully connected layer weights.
        
        Args:
            transfer_mode: 'partial' (transfer what fits), 'reinit' (reinitialize), 'freeze_features' (freeze conv, retrain FC)
        """
        print("\n" + "="*80)
        print(f"ADAPTING FC LAYERS (mode={transfer_mode})")
        print("="*80)
        
        if transfer_mode == 'reinit':
            print("Reinitializing FC layers (keeping conv weights)")
            return ukbb_model
        
        fc_keys = [k for k in self.epic_model_state.keys() 
                   if any(x in k for x in ['fc_shared', 'disease_outputs', 'attention'])]
        
        adapted_count = 0
        for epic_key in fc_keys:
            if epic_key not in ukbb_model.state_dict():
                continue
            
            try:
                epic_param = self.epic_model_state[epic_key]
                ukbb_param = ukbb_model.state_dict()[epic_key]
                
                if epic_param.shape == ukbb_param.shape:
                    ukbb_model.state_dict()[epic_key][:] = epic_param
                    adapted_count += 1
                else:
                    print(f"  Skipping {epic_key}: shape mismatch {epic_param.shape} vs {ukbb_param.shape}")
            except Exception as e:
                print(f"  Warning: Could not adapt {epic_key}: {str(e)}")
        
        print(f"✓ Adapted {adapted_count} FC layer parameters")
        return ukbb_model
    
    def transfer_weights(self, ukbb_model, epic_num_snps, ukbb_num_snps, 
                        transfer_mode='full', freeze_conv=False):
        """
        Main method to transfer all weights from EPIC to UKBB model.
        
        Args:
            ukbb_model: Target UKBB model
            epic_num_snps: Number of SNPs in EPIC dataset
            ukbb_num_snps: Number of SNPs in UKBB dataset
            transfer_mode: 'full', 'conv_only', 'partial'
            freeze_conv: Freeze convolutional layers after transfer
        
        Returns:
            Updated UKBB model with EPIC weights
        """
        print("\n" + "="*80)
        print("STARTING TRANSFER LEARNING: EPIC → UKBB")
        print("="*80)
        print(f"EPIC model SNPs: {epic_num_snps}")
        print(f"UKBB model SNPs: {ukbb_num_snps}")
        print(f"Transfer mode: {transfer_mode}")
        
        # Step 1: Adapt first layer (critical for genomic input)
        ukbb_model = self.adapt_first_layer(ukbb_model, epic_num_snps, ukbb_num_snps)
        
        # Step 2: Adapt remaining conv layers
        if transfer_mode in ['full', 'conv_only']:
            ukbb_model = self.adapt_remaining_layers(ukbb_model)
            ukbb_model = self.adapt_mamba_layers(ukbb_model)
        
        # Step 3: Adapt FC layers
        if transfer_mode == 'full':
            ukbb_model = self.adapt_fc_layers(ukbb_model, transfer_mode='partial')
        
        # Step 4: Optionally freeze conv layers
        if freeze_conv:
            self._freeze_conv_layers(ukbb_model)
        
        print("\n" + "="*80)
        print("✓ TRANSFER LEARNING COMPLETE")
        print("="*80)
        
        return ukbb_model
    
    def _freeze_conv_layers(self, model):
        """Freeze all convolutional layers for fine-tuning"""
        frozen_params = 0
        for name, param in model.named_parameters():
            if 'conv_layers' in name or 'mamba' in name:
                param.requires_grad = False
                frozen_params += 1
        
        print(f"\n✓ Froze {frozen_params} parameters in conv/mamba layers")
    
    def print_weight_statistics(self, model):
        """Print statistics about transferred weights"""
        print("\n" + "="*80)
        print("TRANSFERRED WEIGHT STATISTICS")
        print("="*80)
        
        layer_stats = {}
        for name, param in model.named_parameters():
            layer_type = name.split('.')[0]
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'count': 0, 'mean': [], 'std': []}
            
            layer_stats[layer_type]['count'] += param.numel()
            layer_stats[layer_type]['mean'].append(param.data.mean().item())
            layer_stats[layer_type]['std'].append(param.data.std().item())
        
        for layer_type, stats in layer_stats.items():
            mean_val = np.mean(stats['mean'])
            std_val = np.mean(stats['std'])
            print(f"{layer_type:20s}: {stats['count']:10,} params | mean={mean_val:8.4f}, std={std_val:8.4f}")


def load_ukbb_model_with_epic_weights(ukbb_model, epic_checkpoint_path, 
                                     snp_mapping=None, epic_num_snps=None, 
                                     ukbb_num_snps=None, **transfer_kwargs):
    """
    Convenience function to load EPIC weights into UKBB model.
    
    Usage:
        model = load_ukbb_model_with_epic_weights(
            ukbb_model=model,
            epic_checkpoint_path='/path/to/epic_checkpoint.pt',
            snp_mapping='/path/to/snp_mapping.json',
            epic_num_snps=100000,
            ukbb_num_snps=150000,
            transfer_mode='full',
            freeze_conv=False
        )
    """
    adapter = EPICtoUKBBWeightAdapter(epic_checkpoint_path, snp_mapping)
    
    if snp_mapping and isinstance(snp_mapping, str):
        adapter.load_snp_mapping(snp_mapping)
    
    ukbb_model = adapter.transfer_weights(
        ukbb_model, epic_num_snps, ukbb_num_snps, **transfer_kwargs
    )
    
    adapter.print_weight_statistics(ukbb_model)
    
    return ukbb_model