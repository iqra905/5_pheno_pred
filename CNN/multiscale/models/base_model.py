# multiscale/models/base_model.py

import torch
import torch.nn as nn
from .attention import DiseaseSpecificAttention, SeparateDiseaseHead
from .convolution import MultiScaleConvBlock, ParallelMultiScaleConvBlock
from .transformer import GenomicTransformerBlock


class MultilabelGenotypeModel(nn.Module):
    """Main multilabel genotype prediction model"""
    
    def __init__(self, input_size, num_diseases, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True, num_covariates=10, 
                 use_pooling=True, pool_size=16, pool_type="max",
                 use_multi_scale=True, use_disease_attention=True, use_separate_heads=True, 
                 attention_heads=8, attention_dim=256, multi_scale_kernels=None, multi_scale_strides=None,
                 multi_scale_fusion="cross_scale", multi_scale_mode="progressive", hardcoded_kernels=None, 
                 hardcoded_strides=None, use_pointwise_conv=False, pointwise_channels=16,
                 use_transformer=False, transformer_layers=2, transformer_heads=4, transformer_dim=256, 
                 transformer_ff_dim=1024, transformer_dropout=0.1, use_positional_encoding=True, max_seq_len=10000,
                 init_from_pretrained=False, pretrained_model_name="WinKawaks/vit-small-patch16-224",
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices=""):

        super(MultilabelGenotypeModel, self).__init__()
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.num_diseases = num_diseases
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.use_multi_scale = use_multi_scale
        self.use_disease_attention = use_disease_attention
        self.use_separate_heads = use_separate_heads
        self.use_pointwise_conv = use_pointwise_conv
        self.pointwise_channels = pointwise_channels

        # Store transformer parameters
        self.use_transformer = use_transformer
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout = transformer_dropout
        self.use_positional_encoding = use_positional_encoding
        self.max_seq_len = max_seq_len

        # Store pretrained initialization parameters
        self.init_from_pretrained = init_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.init_layers_fraction = init_layers_fraction
        self.layer_init_strategy = layer_init_strategy
        self.custom_layer_indices = custom_layer_indices

        # Store multi-scale parameters
        self.multi_scale_kernels = multi_scale_kernels if multi_scale_kernels is not None else [15, 63, 255]
        self.multi_scale_strides = multi_scale_strides if multi_scale_strides is not None else [4, 16, 64]
        self.multi_scale_fusion = multi_scale_fusion
        self.multi_scale_mode = multi_scale_mode
        self.hardcoded_kernels = hardcoded_kernels
        self.hardcoded_strides = hardcoded_strides
        
        # Store conv_channels for later reference
        self.conv_channels = conv_channels
        
        # Store input size for sequence length calculations
        self.input_size = input_size

        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender
        if use_bmi:
            self.total_covariates += 1  # BMI

        # Create convolutional layers based on architecture choice
        print(f"  Creating convolutional layers...")
        print(f"  Input sequence length: {input_size:,}")
        
        if self.use_multi_scale:
            print(f"  Multi-scale processing enabled with {self.multi_scale_fusion} fusion")
            print(f"  Multi-scale mode: {self.multi_scale_mode}")
            print(f"  Multi-scale Pool Type: {self.pool_type}")
            print(f"  Pointwise convolution: {'Enabled' if use_pointwise_conv else 'Disabled'}")
            if use_pointwise_conv:
                print(f"  Pointwise channels: {pointwise_channels}")
            
            if self.multi_scale_fusion == "cross_scale":
                print(f"  Using cross-scale fusion strategy")
                conv_layers = self._create_multi_scale_conv_layers_cross_scale(conv_channels, kernel_sizes, stride, dropout_rate, act)
            else:  # parallel
                print(f"  Using parallel processing strategy with independent channels")
                conv_layers = self._create_multi_scale_conv_layers_parallel(conv_channels, kernel_sizes, stride, dropout_rate, act)
            
            # Update the final conv output channels accounting for multi-scale concatenation and pointwise conv
            if use_pointwise_conv:
                final_conv_channels = len(self.multi_scale_kernels) * pointwise_channels
            else:
                final_conv_channels = len(self.multi_scale_kernels) * conv_channels[-1]
        else:
            print(f"  Using standard single-scale convolution")
            conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)
            final_conv_channels = conv_channels[-1]

        # Add final pooling layer to the conv_layers sequential if enabled
        if self.use_pooling:
            if pool_type.lower() == "max":
                pooling_layer = nn.AdaptiveMaxPool1d(pool_size)
                print(f"Adding final AdaptiveMaxPool1d with pool_size={pool_size} to conv_layers")
            elif pool_type.lower() == "avg":
                pooling_layer = nn.AdaptiveAvgPool1d(pool_size)
                print(f"Adding final AdaptiveAvgPool1d with pool_size={pool_size} to conv_layers")
            else:
                raise ValueError(f"Invalid pool_type: {pool_type}. Must be 'max' or 'avg'")
            
            # Create a new sequential that includes the final pooling layer
            conv_layers_with_pool = nn.Sequential(*list(conv_layers), pooling_layer)
            self.conv_layers = conv_layers_with_pool
        else:
            print("Not using final adaptive pooling")
            self.conv_layers = conv_layers
        
        # Calculate output size after convolutions (but before transformer)
        self.conv_output_info = self._get_conv_output_info(input_size)
        conv_output_size = self.conv_output_info['flattened_size']
        conv_seq_len = self.conv_output_info['seq_len']
        conv_channels_out = self.conv_output_info['channels']
        
        print(f"Convolutional output: channels={conv_channels_out}, seq_len={conv_seq_len}, flattened_size={conv_output_size}")
        
        # Add transformer layers if enabled
        if self.use_transformer:
            print(f"\n  Adding Transformer Processing:")
            print(f"  - Using CUSTOM transformer with pretrained weight initialization")
            print(f"  - Transformer layers: {transformer_layers}")
            print(f"  - Transformer heads: {transformer_heads}")
            print(f"  - Transformer dimension: {transformer_dim}")
            print(f"  - Initialize from pretrained: {init_from_pretrained}")
            if init_from_pretrained:
                print(f"  - Pretrained model: {pretrained_model_name}")
            
            print(f"  - Input to transformer: (batch_size, {conv_channels_out}, {conv_seq_len})")
            
            self.transformer_block = GenomicTransformerBlock(
                input_dim=conv_channels_out,
                transformer_dim=transformer_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                ff_dim=transformer_ff_dim,
                dropout=transformer_dropout,
                use_positional_encoding=use_positional_encoding,
                max_seq_len=max_seq_len,
                # Pretrained initialization parameters
                init_from_pretrained=init_from_pretrained,
                pretrained_model_name=pretrained_model_name,
                init_layers_fraction=init_layers_fraction,
                layer_init_strategy=layer_init_strategy,
                custom_layer_indices=custom_layer_indices
            )
            
            # Calculate output size after transformer
            # Transformer output: (batch_size, seq_len, final_output_dim)
            transformer_output_dim = self.transformer_block.final_output_dim
            self.post_transformer_output_size = conv_seq_len * transformer_output_dim
            feature_size_for_fc = self.post_transformer_output_size
            
            print(f"  - Transformer output: (batch_size, {conv_seq_len}, {transformer_output_dim})")
            print(f"  - Flattened size after transformer: {self.post_transformer_output_size:,}")
        else:
            print(f"\n  Transformer processing: Disabled")
            self.transformer_block = None
            feature_size_for_fc = conv_output_size
        
        # Disease-specific attention mechanism
        if self.use_disease_attention:
            self.attention_input_dim = feature_size_for_fc
            self.attention_proj = nn.Linear(self.attention_input_dim, attention_dim)
            self.disease_attention = DiseaseSpecificAttention(attention_dim, num_diseases, attention_heads)
            self.attention_output_dim = attention_dim
            print(f"Using disease-specific attention with {attention_heads} heads and {attention_dim} dimensions")
        else:
            self.attention_output_dim = feature_size_for_fc
            
        # Shared feature layers
        if self.use_separate_heads:
            shared_layers = []
            in_features = self.attention_output_dim + self.total_covariates
            
            for i, out_features in enumerate(fc_layers[:-1]):
                shared_layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    self._get_activation(act),
                    nn.Dropout(dropout_rate)
                ])
                in_features = out_features
            
            self.fc_shared = nn.Sequential(*shared_layers)
            shared_output_dim = in_features
            
            head_hidden_dims = [fc_layers[-1]] if len(fc_layers) > 1 else [64]
            self.disease_heads = nn.ModuleList([
                SeparateDiseaseHead(shared_output_dim, head_hidden_dims, dropout_rate, act)
                for _ in range(num_diseases)
            ])
            print(f"Using separate disease heads with shared feature dimension: {shared_output_dim}")
            
        else:
            fc_layers_list = []
            in_features = self.attention_output_dim + self.total_covariates
            
            for i, out_features in enumerate(fc_layers):
                fc_layers_list.extend([
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    self._get_activation(act),
                    nn.Dropout(dropout_rate)
                ])
                in_features = out_features
            
            self.fc_shared = nn.Sequential(*fc_layers_list)
            self.disease_outputs = nn.Linear(in_features, num_diseases)
        
        self._print_architecture_summary()

    def _print_architecture_summary(self):
        """Print model configuration summary"""
        architecture_info = []
        architecture_info.append(f"Multi-scale convolutions: {self.use_multi_scale}")
        architecture_info.append(f"Final pointwise convolutions: {self.use_pointwise_conv}")
        if self.use_pointwise_conv:
            architecture_info.append(f"Final pointwise channels: {self.pointwise_channels}") 
        
        # Transformer info
        architecture_info.append(f"Transformer layers: {self.use_transformer}")
        if self.use_transformer:
            architecture_info.append(f"Custom transformer: {self.transformer_layers} layers, {self.transformer_heads} heads, dim={self.transformer_dim}")
            architecture_info.append(f"Initialize from pretrained: {self.init_from_pretrained}")
            if self.init_from_pretrained:
                architecture_info.append(f"Pretrained model: {self.pretrained_model_name}")
        
        architecture_info.append(f"Disease-specific attention: {self.use_disease_attention}")
        architecture_info.append(f"Separate disease heads: {self.use_separate_heads}")
        architecture_info.append(f"Using PC's: {self.use_covariates}, age: {self.use_age}, gender: {self.use_gender}, BMI:{self.use_bmi}")
        architecture_info.append(f"Using final pooling: {self.use_pooling} ({self.pool_type} pool, size={self.pool_size})" if self.use_pooling else "Using final pooling: False")
        architecture_info.append(f"Multi-scale internal pooling: {self.pool_type} (for length standardization)" if self.use_multi_scale else "Internal pooling: N/A")
        
        print(f"\nEnhanced MultilabelGenotypeModel initialized:")
        for info in architecture_info:
            print(f"  - {info}")
        
        print(f"  - Input size: {self.input_size:,} SNPs")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _calculate_conv_output_length(self, input_length, kernel_size, stride, padding):
        """Helper method to calculate output sequence length after convolution"""
        return (input_length + 2 * padding - kernel_size) // stride + 1

    def _print_sequence_progression(self, layer_idx, branch_idx, branch_name, input_length, kernel_size, stride, padding, channels_in, channels_out, is_final_layer=False):
        """Helper method to print sequence length progression"""
        output_length = self._calculate_conv_output_length(input_length, kernel_size, stride, padding)

        if self.use_pointwise_conv and is_final_layer:
            pointwise_out = self.pointwise_channels
            print(f"      Branch {branch_idx} ({branch_name}): "
                  f"sequence {input_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride}, padding={padding}) "
                  f"channels {channels_in} → {channels_out} → {pointwise_out} (final pointwise)")
        else:
            print(f"      Branch {branch_idx} ({branch_name}): "
                  f"sequence {input_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride}, padding={padding}) "
                  f"channels {channels_in} → {channels_out}")
        return output_length

    def _get_layer_kernels_and_strides(self, layer_idx):
        """Get kernels and strides for a specific layer based on mode"""
        if self.multi_scale_mode == "hardcoded":
            # Use hardcoded values
            return self.hardcoded_kernels[layer_idx], self.hardcoded_strides[layer_idx]
        else:
            # Use progressive reduction (original behavior)
            if layer_idx == 0:
                return self.multi_scale_kernels.copy(), self.multi_scale_strides.copy()
            else:
                scale_kernels = []
                scale_strides = []
                
                for j in range(len(self.multi_scale_kernels)):
                    # Calculate kernel: original_kernel // (2^i)
                    new_kernel = max(1, self.multi_scale_kernels[j] // (2 ** layer_idx))
                    # Ensure kernel is odd for proper padding
                    if new_kernel % 2 == 0:
                        new_kernel += 1
                    
                    # Calculate stride: original_stride // (2^i)
                    new_stride = max(1, self.multi_scale_strides[j] // (2 ** layer_idx))
                    
                    scale_kernels.append(new_kernel)
                    scale_strides.append(new_stride)
                
                return scale_kernels, scale_strides

    def _create_multi_scale_conv_layers_cross_scale(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create multi-scale convolutional layers with cross-scale fusion"""
        layers = nn.ModuleList()
        
        print(f"  Sequence length progression:")
        
        current_length = self.input_size
        
        for i in range(len(conv_channels)):
            print(f"\n  Layer {i}:")
            print(f"    Input sequence length: {current_length:,}")
            
            # Determine if this is the final layer
            is_final_layer = (i == len(conv_channels) - 1)
            
            # Calculate input channels considering multi-scale concatenation and pointwise conv
            if i == 0:
                in_channels = self.input_channels
            else:
                # Previous layer output channels depend on whether pointwise was applied to previous layer
                prev_is_final = (i-1 == len(conv_channels) - 1)
                if self.use_pointwise_conv and prev_is_final:
                    in_channels = len(self.multi_scale_kernels) * self.pointwise_channels
                else:
                    in_channels = len(self.multi_scale_kernels) * conv_channels[i-1]
            
            # Each branch gets the FULL conv_channels[i] value
            out_channels = conv_channels[i]
            
            # Get kernels and strides for this layer
            scale_kernels, scale_strides = self._get_layer_kernels_and_strides(i)
            
            if self.multi_scale_mode == "progressive":
                if i == 0:
                    print(f"    Genomic multi-scale kernels={scale_kernels}, strides={scale_strides}")
                    branch_names = ['Local LD blocks', 'Gene-level regions', 'Long-range domains']
                else:
                    print(f"    Progressive multi-scale kernels={scale_kernels}, strides={scale_strides}")
                    print(f"    Reduction factor: kernels ÷ {2**i}, strides ÷ {2**i}")
                    branch_names = ['Local', 'Gene', 'Domain']
            else:  # hardcoded
                print(f"    Hardcoded multi-scale kernels={scale_kernels}, strides={scale_strides}")
                branch_names = [f'Branch_{j}' for j in range(len(scale_kernels))]
            
            # Calculate output lengths for each branch
            branch_outputs = []
            for j, (kernel_size, stride_val) in enumerate(zip(scale_kernels, scale_strides)):
                padding = kernel_size // 2
                branch_name = branch_names[j] if j < len(branch_names) else f'Scale_{j}'
                output_length = self._print_sequence_progression(
                    i, j, branch_name, current_length, kernel_size, stride_val, padding, in_channels, out_channels, is_final_layer
                )
                branch_outputs.append(output_length)
            
            # After multi-scale block, length is minimum of all branches
            final_length = branch_outputs[1] if len(branch_outputs) >=2 else branch_outputs[0]
            print(f"    After concatenation: sequence length = {final_length:,}")
            
            if self.use_pointwise_conv and is_final_layer:
                print(f"    Output channels: {len(scale_kernels)} * {self.pointwise_channels} = {len(scale_kernels) * self.pointwise_channels} (after final pointwise)")
            else:
                print(f"    Output channels: {len(scale_kernels)} * {out_channels} = {len(scale_kernels) * out_channels}")
            print(f"    Internal {self.pool_type} pooling applied for length standardization")
            
            multi_scale_block = MultiScaleConvBlock(
                in_channels, out_channels, scale_kernels, scale_strides, act, dropout_rate, self.pool_type,
                 self.use_pointwise_conv, self.pointwise_channels, is_final_layer
            )
            layers.append(multi_scale_block)
            
            # Update current length for next layer
            current_length = final_length
        
        print(f"\n  Final sequence length after all conv layers: {current_length:,}")
        return nn.Sequential(*layers)
    
    def _create_multi_scale_conv_layers_parallel(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create parallel multi-scale convolutional layers in sequential manner"""
        layers = nn.ModuleList()
        
        print(f"  Using PARALLEL fusion strategy (Sequential Implementation)")
        print(f"  Sequence length progression:")
        
        # Track sequence lengths for each branch independently
        branch_lengths = [self.input_size] * len(self.multi_scale_kernels)
        branch_names = ['Local LD blocks', 'Gene-level regions', 'Long-range domains']
        
        for i in range(len(conv_channels)):
            print(f"\n  Layer {i}:")

            # Determine if this is the final layer
            is_final_layer = (i == len(conv_channels) - 1)
            
            # Input channels are same as specified (no concatenation from previous layer)
            in_channels = self.input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            
            # Get kernels and strides for this layer
            scale_kernels, scale_strides = self._get_layer_kernels_and_strides(i)
            
            if self.multi_scale_mode == "progressive":
                if i == 0:
                    print(f"    Genomic multi-scale (independent channels)")
                else:
                    print(f"    Progressive parallel multi-scale (independent channels)")
                    print(f"    Reduction factor: kernels ÷ {2**i}, strides ÷ {2**i}")
            else:  # hardcoded
                print(f"    Hardcoded parallel multi-scale (independent channels)")
            
            if self.use_pointwise_conv and is_final_layer:
                print(f"    Each branch: {in_channels} → {out_channels} → {self.pointwise_channels} channels (final pointwise)")
            else:
                print(f"    Each branch: {in_channels} → {out_channels} channels")
            
            # Calculate sequence progression for each branch
            for j, (kernel_size, stride_val) in enumerate(zip(scale_kernels, scale_strides)):
                padding = kernel_size // 2
                branch_name = branch_names[j] if j < len(branch_names) else f'Branch_{j}'
                
                new_length = self._print_sequence_progression(
                    i, j, branch_name, branch_lengths[j], kernel_size, stride_val, padding, in_channels, out_channels, is_final_layer
                )
                branch_lengths[j] = new_length
            
            if is_final_layer:
                # Final layer: show concatenation info
                final_length = branch_lengths[1] if len(branch_lengths) >=2 else branch_lengths[0]
                print(f"    Final concatenation: min sequence length = {final_length:,}")
                
                if self.use_pointwise_conv:
                    print(f"    Total output channels: {len(self.multi_scale_kernels)} * {self.pointwise_channels} = {len(self.multi_scale_kernels) * self.pointwise_channels} (after final pointwise)")
                else:
                    print(f"    Total output channels: {len(self.multi_scale_kernels)} * {out_channels} = {len(self.multi_scale_kernels) * out_channels}")
                print(f"    Final layer {self.pool_type} pooling applied for length standardization")
            
            # Create parallel multi-scale block for this layer
            parallel_block = ParallelMultiScaleConvBlock(
                in_channels, out_channels, scale_kernels, scale_strides, act, dropout_rate, i, is_final_layer, self.pool_type,
                self.use_pointwise_conv, self.pointwise_channels
            )
            layers.append(parallel_block)
        
        final_sequence_length = min(branch_lengths)
        print(f"\n  Final sequence length after all conv layers: {final_sequence_length:,}")
        
        # Return as Sequential, just like cross-scale mode
        return nn.Sequential(*layers)

    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create traditional sequential convolutional layers"""
        layers = []
        current_length = self.input_size
        
        print(f"  Standard convolution sequence length progression:")
        
        for i in range(len(conv_channels)):
            in_channels = self.input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            kernel_size = kernel_sizes[i]
            stride_val = stride[i]
            padding = kernel_size // 2
            
            # Calculate output length
            output_length = self._calculate_conv_output_length(current_length, kernel_size, stride_val, padding)
            
            print(f"    Layer {i}: sequence {current_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride_val}, padding={padding}) "
                  f"channels {in_channels} → {out_channels}")
            
            layers.append(nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,  
                                   kernel_size=kernel_size,
                                   stride=stride_val,
                                   padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation(act))
            
            current_length = output_length
        
        print(f"  Final sequence length after standard conv layers: {current_length:,}")
        return nn.Sequential(*layers)

    def _get_conv_output_info(self, input_size):
        """Calculate output dimensions after convolution layers"""
        try:
            x = torch.randn(1, 3, input_size, dtype=torch.float32)
            print(f"  - Input to conv layers: {x.shape}")
            
            x = self.conv_layers(x)
            print(f"  - Output from conv layers: {x.shape}")
            
            batch_size, channels, seq_len = x.shape
            flattened_size = x.numel() // x.size(0)
            
            return {
                'channels': channels,
                'seq_len': seq_len,
                'flattened_size': flattened_size
            }
            
        except Exception as e:
            print(f"Error in _get_conv_output_info: {e}")
            raise e

    def _get_activation(self, name):
        """Get activation function by name"""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.01)
        elif name == 'rrelu':
            return nn.RReLU(0.125, 0.3333)
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise NotImplementedError("Activation function not implemented.")

    def forward(self, x, covariates=None):
        # Input: x shape [batch_size, n_snps, 3]
        x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
        
        # Convolutional processing (includes final pooling if enabled)
        x = self.conv_layers(x)  # Shape: (batch_size, channels, seq_len)
        
        # Transformer processing if enabled
        if self.use_transformer:
            # Transform: (batch_size, channels, seq_len) -> (batch_size, seq_len, transformer_dim)
            x = self.transformer_block(x)
            
            # Flatten transformer output for fully connected layers
            x = x.view(x.size(0), -1)  # [batch_size, seq_len * transformer_dim]
        else:
            # Flatten conv output for fully connected layers
            x = x.view(x.size(0), -1)  # [batch_size, flattened_features]
        
        # Disease-specific attention processing
        if self.use_disease_attention:
            # Project to attention dimension and add sequence dimension
            x_proj = self.attention_proj(x)  # [batch_size, attention_dim]
            x_seq = x_proj.unsqueeze(1)  # [batch_size, 1, attention_dim]
            
            # Apply disease-specific attention
            x_attended = self.disease_attention(x_seq)  # [batch_size, num_diseases, attention_dim]
            
            # For now, use mean pooling across diseases for shared processing
            x = torch.mean(x_attended, dim=1)  # [batch_size, attention_dim]
        
        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Shared feature processing
        shared_features = self.fc_shared(x)
        
        if self.use_separate_heads:
            # Use separate heads for each disease
            if self.use_disease_attention:
                # Use disease-specific features for each head
                disease_outputs = []
                for i, head in enumerate(self.disease_heads):
                    # Get disease-specific features
                    disease_features = x_attended[:, i, :]  # [batch_size, attention_dim]
                    
                    # Concatenate with covariates
                    if covariates is not None and self.total_covariates > 0:
                        disease_input = torch.cat([disease_features, covariates], dim=1)
                    else:
                        disease_input = disease_features
                    
                    # Process through shared layers first
                    disease_shared = self.fc_shared(disease_input)
                    
                    # Then through disease-specific head
                    disease_output = head(disease_shared)
                    disease_outputs.append(disease_output)
                
                return torch.cat(disease_outputs, dim=1)
            else:
                # Use shared features for all heads
                disease_outputs = []
                for head in self.disease_heads:
                    disease_output = head(shared_features)
                    disease_outputs.append(disease_output)
                
                return torch.cat(disease_outputs, dim=1)
        else:
            # Traditional shared output layer
            return self.disease_outputs(shared_features)