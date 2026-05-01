import torch
import torch.nn as nn
import numpy as np
import os

class EPICtoUKBBWeightAdapter:
    """
    Simplified weight adapter for EPIC → UKBB transfer learning.
    Assumes both models use the SAME SNP indices (filtered by dataloader).
    """
    
    def __init__(self, epic_checkpoint_path):
        """
        Args:
            epic_checkpoint_path: Path to EPIC model checkpoint
        """
        self.epic_checkpoint = torch.load(epic_checkpoint_path, map_location='cpu', weights_only=False)
        self.epic_model_state = self.epic_checkpoint.get('model_state_dict', self.epic_checkpoint)
        
        print(f"✓ Loaded EPIC checkpoint from: {epic_checkpoint_path}")
        print(f"✓ EPIC model has {sum(p.numel() for p in self.epic_checkpoint.get('model_state_dict', self.epic_checkpoint).values()):,} parameters")
    
    def transfer_weights(self, ukbb_model, freeze_conv=False, freeze_mamba=False):
        """
        Transfer weights layer-by-layer from EPIC to UKBB.
        Both models assume SAME SNP indices (already filtered).
        
        Args:
            ukbb_model: Target UKBB model
            freeze_conv: Freeze convolutional layers after transfer
            freeze_mamba: Freeze Mamba layers after transfer
        
        Returns:
            Updated UKBB model with EPIC weights
        """
        print("\n" + "="*80)
        print("STARTING TRANSFER LEARNING: EPIC → UKBB")
        print("="*80)
        
        ukbb_state = ukbb_model.state_dict()
        transferred = 0
        skipped = 0
        
        print(f"\nTransferring {len(self.epic_model_state)} parameters...")
        
        for epic_key, epic_param in self.epic_model_state.items():
            # Try exact match first
            if epic_key in ukbb_state:
                ukbb_param = ukbb_state[epic_key]
                
                # Check shape compatibility
                if epic_param.shape == ukbb_param.shape:
                    try:
                        ukbb_state[epic_key].copy_(epic_param)
                        transferred += 1
                    except Exception as e:
                        print(f"  Error copying {epic_key}: {str(e)}")
                        skipped += 1
                else:
                    print(f"  ⚠ Shape mismatch: {epic_key}")
                    print(f"    EPIC: {epic_param.shape} → UKBB: {ukbb_param.shape}")
                    skipped += 1
            else:
                skipped += 1
        
        # Load the updated state
        ukbb_model.load_state_dict(ukbb_state, strict=False)
        
        print(f"\n✓ Transfer complete:")
        print(f"  - Transferred: {transferred} parameters")
        print(f"  - Skipped: {skipped} parameters")
        
        # Optionally freeze layers
        if freeze_conv:
            self._freeze_layers(ukbb_model, ['conv_layers'])
        if freeze_mamba:
            self._freeze_layers(ukbb_model, ['mamba_block'])
        
        print("\n" + "="*80)
        print("✓ WEIGHT TRANSFER COMPLETE")
        print("="*80)
        
        return ukbb_model
    
    def _freeze_layers(self, model, layer_names):
        """Freeze specific layers"""
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                frozen_count += 1
        
        print(f"✓ Froze {frozen_count} parameters in {layer_names}")
    
    def print_weight_statistics(self, model):
        """Print before/after weight statistics"""
        print("\n" + "="*80)
        print("TRANSFERRED WEIGHT STATISTICS")
        print("="*80)
        
        for name, param in model.named_parameters():
            if param.numel() > 0:
                print(f"{name:50s} | shape: {str(param.shape):30s} | mean: {param.mean():8.4f}, std: {param.std():8.4f}")


def load_ukbb_model_with_epic_weights(ukbb_model, epic_checkpoint_path, 
                                     freeze_conv=False, freeze_mamba=False):
    """
    Convenience function to load EPIC weights into UKBB model.
    
    Usage in main script:
        model = load_ukbb_model_with_epic_weights(
            ukbb_model=model,
            epic_checkpoint_path=args.epic_checkpoint,
            freeze_conv=args.freeze_conv_layers,
            freeze_mamba=args.freeze_mamba_layers
        )
    """
    adapter = EPICtoUKBBWeightAdapter(epic_checkpoint_path)
    ukbb_model = adapter.transfer_weights(ukbb_model, freeze_conv, freeze_mamba)
    #adapter.print_weight_statistics(ukbb_model)
    return ukbb_model