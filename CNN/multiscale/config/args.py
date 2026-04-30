# multiscale/config/args.py

import argparse


def parse_int_list(s):
    return [int(x) for x in s.split(',')]


def parse_str_list(s):
    return [x.strip() for x in s.split(',')]


def parse_nested_int_list(s):
    """Parse nested lists like '15,63,255;7,31,127;3,15,63' into [[15,63,255], [7,31,127], [3,15,63]]"""
    if not s or s.lower() == 'none':
        return None
    layers = s.split(';')
    return [[int(x) for x in layer.split(',')] for layer in layers]


def parse_args():
    parser = argparse.ArgumentParser(description="Multilabel Genotype Model Training")
    
    # Experiment settings
    parser.add_argument("-ID", type=str, default="Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, 
                       default='/mnt/fast/nobackup/users/if00208/5_disease_experiments/CNN/results/5d_multilabel/multiscale',
                       help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str,
                       default='/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy',
                       help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str,
                       default='/mnt/fast/datasets/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx',
                       help="Path to phenotype file")

    # Model and training parameters
    parser.add_argument("-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], 
                       help="Activation function for the model")
    parser.add_argument("-sch", type=str, default="exponential_decay", 
                       choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"],
                       help="Learning rate scheduler")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                       help="Optimizer to use")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")

    # Model architecture
    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[127,31,7], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[64,16,4], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[32,64,128], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers")

    # Enhanced architecture parameters
    parser.add_argument("-use_multi_scale", type=int, default=1, choices=[0, 1], 
                       help="Whether to use multi-scale convolutions (0: no, 1: yes)")
    parser.add_argument("-use_disease_attention", type=int, default=0, choices=[0, 1], 
                       help="Whether to use disease-specific attention (0: no, 1: yes)")
    parser.add_argument("-use_separate_heads", type=int, default=0, choices=[0, 1], 
                       help="Whether to use separate disease heads (0: no, 1: yes)")
    parser.add_argument("-attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("-attention_dim", type=int, default=256, help="Attention dimension")
    
    # Multi-scale configuration
    parser.add_argument("-multi_scale_kernels", type=parse_int_list, default=[15,127], 
                       help="Multi-scale kernel sizes for first layer")
    parser.add_argument("-multi_scale_strides", type=parse_int_list, default=[8,16], 
                       help="Multi-scale strides for first layer")
    parser.add_argument("-multi_scale_fusion", type=str, default="parallel", 
                       choices=["cross_scale", "parallel"], 
                       help="Multi-scale fusion strategy: cross_scale (branches see all scales) or parallel (independent branches)")

    # Multi-scale mode selection
    parser.add_argument("-multi_scale_mode", type=str, default="hardcoded", 
                       choices=["progressive", "hardcoded"], 
                       help="Multi-scale mode: 'progressive' (kernel//2^i, stride//2^i) or 'hardcoded' (explicit values for each layer)")
    
    # Hardcoded multi-scale parameters
    parser.add_argument("-hardcoded_kernels", type=parse_nested_int_list, default='16,128,1024;16,64,512;16,32,256',
                       help="Hardcoded kernel sizes for all layers and branches")
    parser.add_argument("-hardcoded_strides", type=parse_nested_int_list, default='16,16,16;16,16,16;16,16,16',
                       help="Hardcoded stride values for all layers and branches")

    # Pointwise convolution parameters
    parser.add_argument("-use_pointwise_conv", type=int, default=0, choices=[0, 1], 
                       help="Whether to use pointwise (1x1) convolution after each branch before concatenation")
    parser.add_argument("-pointwise_channels", type=int, default=32, 
                       help="Number of output channels for pointwise convolution")

    # Data-specific parameters
    parser.add_argument("-cov", type=int, default=0, choices=[0, 1], 
                       help="Whether to include covariates in the model")
    parser.add_argument("-use_age", type=int, default=0, choices=[0, 1], 
                       help="Whether to include age in covariates")
    parser.add_argument("-use_gender", type=int, default=0, choices=[0, 1], 
                       help="Whether to include gender in covariates")
    parser.add_argument("-use_bmi", type=int, default=0, choices=[0, 1], 
                       help="Whether to include BMI in covariates")

    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization arguments
    parser.add_argument("-norm_age", type=str, default="none", 
                       choices=["none", "standard", "minmax", "robust", "quantile", "power"], 
                       help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", 
                       choices=["none", "standard", "minmax", "robust", "quantile", "power"], 
                       help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], 
                       help="Normalization method for gender")
    parser.add_argument("-norm_bmi", type=str, default="none", 
                       choices=["none", "standard", "minmax", "robust", "quantile", "power"], 
                       help="Normalization method for BMI")
    parser.add_argument("-disease_labels", type=parse_str_list, default="pros01,panca,crc,breacancer,t2dm", 
                       help="Comma-separated list of disease column names")

    # Pooling parameters
    parser.add_argument("-use_pooling", type=int, default=1, choices=[0, 1], 
                       help="Whether to use Pooling after convolution layers")
    parser.add_argument("-pool_size", type=int, default=512, 
                       help="Size of the adaptive pooling output")
    parser.add_argument("-pool_type", type=str, default="max", choices=["max", "avg"], 
                       help="Type of adaptive pooling")
    
    # Transformer-related arguments
    parser.add_argument("-use_transformer", type=int, default=1, choices=[0, 1], 
                       help="Whether to use transformer layers after convolution")
    parser.add_argument("-transformer_layers", type=int, default=2, 
                       help="Number of transformer encoder layers")
    parser.add_argument("-transformer_heads", type=int, default=4, 
                       help="Number of attention heads in transformer")
    parser.add_argument("-transformer_dim", type=int, default=384, 
                       help="Transformer model dimension (d_model)")
    parser.add_argument("-transformer_ff_dim", type=int, default=1024, 
                       help="Transformer feedforward dimension")
    parser.add_argument("-transformer_dropout", type=float, default=0.1, 
                       help="Transformer dropout rate")
    parser.add_argument("-use_positional_encoding", type=int, default=1, choices=[0, 1], 
                       help="Whether to use positional encoding in transformer")
    parser.add_argument("-max_seq_len", type=int, default=10000, 
                       help="Maximum sequence length for positional encoding")

    # Pretrained weight initialization arguments
    parser.add_argument("-init_from_pretrained", type=int, default=1, choices=[0, 1], 
                       help="Whether to initialize transformer weights from pretrained model")
    parser.add_argument("-pretrained_model_name", type=str, default="WinKawaks/vit-small-patch16-224", 
                       help="Name of pretrained model for weight initialization")
    parser.add_argument("-init_layers_fraction", type=float, default=1.0, 
                       help="Fraction of transformer layers to initialize from pretrained")

    # Layer selection strategy arguments
    parser.add_argument("-layer_init_strategy", type=str, default="middle", 
                       choices=["first", "middle", "last", "random", "custom"], 
                       help="Strategy for selecting which pretrained layers to use")
    parser.add_argument("-custom_layer_indices", type=str, default="4,6", 
                       help="Comma-separated list of pretrained layer indices to use")
    
    # Checkpoint-related parameters
    parser.add_argument("-resume", type=int, default=1, choices=[0, 1], 
                       help="Whether to resume from checkpoint if available")
    parser.add_argument("-keep_checkpoints", type=int, default=1, 
                       help="Number of recent checkpoints to keep")
    
    return parser.parse_args()


def validate_hardcoded_parameters(args):
    """Validate hardcoded multi-scale parameters"""
    if args.multi_scale_mode == "hardcoded":
        if args.hardcoded_kernels is None or args.hardcoded_strides is None:
            raise ValueError("When using hardcoded multi-scale mode, both --hardcoded_kernels and --hardcoded_strides must be provided")
        
        if len(args.hardcoded_kernels) != len(args.conv_channels):
            raise ValueError(f"Number of hardcoded kernel layers ({len(args.hardcoded_kernels)}) must match number of conv_channels ({len(args.conv_channels)})")
        
        if len(args.hardcoded_strides) != len(args.conv_channels):
            raise ValueError(f"Number of hardcoded stride layers ({len(args.hardcoded_strides)}) must match number of conv_channels ({len(args.conv_channels)})")
        
        # Check that all layers have the same number of branches
        num_branches = len(args.hardcoded_kernels[0])
        for i, layer_kernels in enumerate(args.hardcoded_kernels):
            if len(layer_kernels) != num_branches:
                raise ValueError(f"All layers must have the same number of branches. Layer {i} has {len(layer_kernels)} branches, expected {num_branches}")
        
        for i, layer_strides in enumerate(args.hardcoded_strides):
            if len(layer_strides) != num_branches:
                raise ValueError(f"All layers must have the same number of branches. Layer {i} has {len(layer_strides)} stride values, expected {num_branches}")
        
        # Update multi_scale_kernels and strides to match the first layer of hardcoded values
        args.multi_scale_kernels = args.hardcoded_kernels[0]
        args.multi_scale_strides = args.hardcoded_strides[0]
        
        print(f"Hardcoded multi-scale mode validated:")
        print(f"  - Number of layers: {len(args.hardcoded_kernels)}")
        print(f"  - Number of branches per layer: {num_branches}")
        print(f"  - Hardcoded kernels: {args.hardcoded_kernels}")
        print(f"  - Hardcoded strides: {args.hardcoded_strides}")