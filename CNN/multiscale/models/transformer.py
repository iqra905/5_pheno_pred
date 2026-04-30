# multiscale/models/transformer.py

import math
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer layers"""
    
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GenomicTransformerBlock(nn.Module):
    """Transformer encoder block optimized for genomic sequences with flexible pretrained weight initialization"""
    
    def __init__(self, input_dim, transformer_dim, num_layers=2, num_heads=4, 
                 ff_dim=1024, dropout=0.1, use_positional_encoding=True, max_seq_len=10000,
                 init_from_pretrained=False, pretrained_model_name="WinKawaks/vit-small-patch16-224",
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices=""):
        super(GenomicTransformerBlock, self).__init__()
        
        self.input_dim = input_dim
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.init_from_pretrained = init_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.init_layers_fraction = init_layers_fraction
        
        self.layer_init_strategy = layer_init_strategy
        self.custom_layer_indices = custom_layer_indices
        
        # Project input to transformer dimension if needed
        self.input_projection = None
        if input_dim != transformer_dim:
            self.input_projection = nn.Linear(input_dim, transformer_dim)
            print(f"  - Adding input projection: {input_dim} → {transformer_dim}")
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(transformer_dim, max_seq_len, dropout)
            print(f"  - Adding positional encoding (max_len={max_seq_len})")
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (seq_len, batch_size, features)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.final_output_dim = transformer_dim
        
        # Initialize weights from pretrained model if requested
        if init_from_pretrained:
            self._initialize_from_pretrained()
        
        print(f"GenomicTransformerBlock initialized:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Output dimension: {self.final_output_dim}")
        print(f"  - Transformer dimension: {transformer_dim}")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Feedforward dimension: {ff_dim}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Using positional encoding: {use_positional_encoding}")
        print(f"  - Initialize from pretrained: {init_from_pretrained}")
        if init_from_pretrained:
            print(f"  - Pretrained model: {pretrained_model_name}")
            print(f"  - Layer selection strategy: {layer_init_strategy}")
    
    def _initialize_from_pretrained(self):
        """Initialize transformer weights from pretrained Vision Transformer with flexible layer selection"""
        print(f"\n  Initializing transformer weights from pretrained model: {self.pretrained_model_name}")
        
        try:
            # Load pretrained ViT model
            pretrained_vit = ViTModel.from_pretrained(self.pretrained_model_name)
            pretrained_layers = pretrained_vit.encoder.layer
            
            print(f"  - Loaded pretrained model with {len(pretrained_layers)} layers")
            print(f"  - Pretrained hidden size: {pretrained_vit.config.hidden_size}")
            print(f"  - Pretrained num heads: {pretrained_vit.config.num_attention_heads}")
            print(f"  - Pretrained intermediate size: {pretrained_vit.config.intermediate_size}")
            
            # Calculate how many layers to initialize
            num_layers_to_init = int(self.num_layers * self.init_layers_fraction)
            print(f"  - Initializing {num_layers_to_init} out of {self.num_layers} custom layers")
            
            # Determine which pretrained layers to use
            pretrained_layer_indices = self._select_pretrained_layers(
                len(pretrained_layers), 
                num_layers_to_init
            )
            
            print(f"  - Using pretrained layers: {pretrained_layer_indices}")
            print(f"  - Layer selection strategy: {self.layer_init_strategy}")
            
            self._initialize_without_averaging(pretrained_layers, pretrained_layer_indices)
            
            print(f"  - Weight initialization completed successfully")
            
        except Exception as e:
            print(f"  - Warning: Could not initialize from pretrained model: {e}")
            print(f"  - Continuing with random initialization")

    def _select_pretrained_layers(self, total_pretrained_layers, num_layers_needed):
        """Select which pretrained layers to use based on strategy"""
        
        if self.layer_init_strategy == "first":
            # Use first N layers: [0, 1, 2, ...]
            selected = list(range(min(num_layers_needed, total_pretrained_layers)))
            print(f"    Selected first layers: {selected}")
            return selected
        
        elif self.layer_init_strategy == "middle":
            # Use middle layers
            if total_pretrained_layers < num_layers_needed:
                return list(range(total_pretrained_layers))
            
            # Calculate middle range
            start_idx = (total_pretrained_layers - num_layers_needed) // 2
            end_idx = start_idx + num_layers_needed
            selected = list(range(start_idx, end_idx))
            
            print(f"    Selected middle layers from index {start_idx} to {end_idx-1}: {selected}")
            return selected
        
        elif self.layer_init_strategy == "last":
            # Use last N layers: [..., -3, -2, -1]
            start_idx = max(0, total_pretrained_layers - num_layers_needed)
            selected = list(range(start_idx, total_pretrained_layers))
            
            print(f"    Selected last layers from index {start_idx} to {total_pretrained_layers-1}: {selected}")
            return selected
        
        elif self.layer_init_strategy == "random":
            # Randomly select N layers
            import random
            available_indices = list(range(total_pretrained_layers))
            selected = sorted(random.sample(available_indices, min(num_layers_needed, total_pretrained_layers)))
            
            print(f"    Randomly selected layers: {selected}")
            return selected
        
        elif self.layer_init_strategy == "custom":
            # Use user-specified layers
            if not self.custom_layer_indices:
                print(f"    Warning: custom strategy selected but no indices provided, falling back to first layers")
                return list(range(min(num_layers_needed, total_pretrained_layers)))
            
            try:
                custom_indices = [int(x.strip()) for x in self.custom_layer_indices.split(',')]
                # Validate indices
                valid_indices = [idx for idx in custom_indices if 0 <= idx < total_pretrained_layers]
                
                if len(valid_indices) != len(custom_indices):
                    print(f"    Warning: Some indices were invalid. Using valid indices: {valid_indices}")
                
                # Take only what we need
                selected = valid_indices[:num_layers_needed]
                print(f"    Using custom layer indices: {selected}")
                return selected
                
            except ValueError as e:
                print(f"    Error parsing custom indices: {e}, falling back to first layers")
                return list(range(min(num_layers_needed, total_pretrained_layers)))
        
        else:
            # Fallback to first layers
            print(f"    Unknown strategy '{self.layer_init_strategy}', using first layers")
            return list(range(min(num_layers_needed, total_pretrained_layers)))

    def _initialize_without_averaging(self, pretrained_layers, pretrained_layer_indices):
        """Initialize each custom layer from a single pretrained layer"""
        for i, pretrained_idx in enumerate(pretrained_layer_indices):
            if i < len(self.transformer_encoder.layers):
                print(f"    Initializing custom layer {i} from pretrained layer {pretrained_idx}")
                self._copy_layer_weights(
                    pretrained_layers[pretrained_idx], 
                    self.transformer_encoder.layers[i],
                    layer_idx=i
                )

    def _copy_layer_weights(self, pretrained_layer, custom_layer, layer_idx):
        """Copy weights from pretrained layer to custom layer with intelligent dimension adaptation"""
        print(f"    Initializing layer {layer_idx}")
        
        def adapt_weight_tensor(pretrained_weight, target_shape, weight_name):
            """Intelligently adapt pretrained weights to target shape"""
            pretrained_shape = pretrained_weight.shape
            
            if pretrained_shape == target_shape:
                return pretrained_weight.clone()
            
            # For 2D weights (Linear layer weights)
            if len(target_shape) == 2 and len(pretrained_shape) == 2:
                target_out, target_in = target_shape
                pretrained_out, pretrained_in = pretrained_shape
                
                # Create new tensor with target shape
                adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
                
                # Copy overlapping region
                copy_out = min(target_out, pretrained_out)
                copy_in = min(target_in, pretrained_in)
                
                adapted_weight[:copy_out, :copy_in] = pretrained_weight[:copy_out, :copy_in]
                
                # Initialize remaining weights
                if target_out > pretrained_out or target_in > pretrained_in:
                    # Use Xavier/Glorot initialization for new weights
                    with torch.no_grad():
                        fan_in, fan_out = target_in, target_out
                        std = math.sqrt(2.0 / (fan_in + fan_out))
                        
                        # Initialize new rows (if target_out > pretrained_out)
                        if target_out > pretrained_out:
                            adapted_weight[copy_out:, :copy_in].normal_(0, std)
                        
                        # Initialize new columns (if target_in > pretrained_in)  
                        if target_in > pretrained_in:
                            adapted_weight[:copy_out, copy_in:].normal_(0, std)
                            
                        # Initialize new intersection (if both dimensions expanded)
                        if target_out > pretrained_out and target_in > pretrained_in:
                            adapted_weight[copy_out:, copy_in:].normal_(0, std)
                
                return adapted_weight
                
            # For 1D weights (bias vectors, layer norm weights)
            elif len(target_shape) == 1 and len(pretrained_shape) == 1:
                target_dim = target_shape[0]
                pretrained_dim = pretrained_shape[0]
                
                adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
                
                copy_dim = min(target_dim, pretrained_dim)
                adapted_weight[:copy_dim] = pretrained_weight[:copy_dim]
                
                # Initialize remaining elements (bias usually to 0, others to small random)
                if target_dim > pretrained_dim:
                    if 'bias' in weight_name.lower():
                        adapted_weight[copy_dim:] = 0.0
                    else:
                        adapted_weight[copy_dim:].normal_(0, 0.02)
                
                return adapted_weight
            
            # For unsupported shapes, fall back to Xavier initialization
            else:
                print(f"        ⚠ Unsupported shape adaptation, using Xavier initialization")
                adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
                torch.nn.init.xavier_uniform_(adapted_weight)
                return adapted_weight
        
        def adapt_bias_tensor(pretrained_bias, target_shape, bias_name):
            """Adapt bias tensors with appropriate initialization"""
            if pretrained_bias.shape == target_shape:
                return pretrained_bias.clone()
            
            target_dim = target_shape[0]
            pretrained_dim = pretrained_bias.shape[0]
            
            adapted_bias = torch.zeros(target_shape, dtype=pretrained_bias.dtype, device=pretrained_bias.device)
            
            copy_dim = min(target_dim, pretrained_dim)
            adapted_bias[:copy_dim] = pretrained_bias[:copy_dim]
            
            # Bias initialization for new elements is usually 0
            if target_dim > pretrained_dim:
                adapted_bias[copy_dim:] = 0.0
            
            return adapted_bias
        
        try:
            # Get pretrained attention weights
            pretrained_attn = pretrained_layer.attention.attention
            custom_attn = custom_layer.self_attn
            
            # Adapt and copy Q, K, V weights
            embed_dim = custom_attn.embed_dim
            
            # Get target shapes for Q, K, V projections
            q_target_shape = (embed_dim, embed_dim)
            k_target_shape = (embed_dim, embed_dim)  
            v_target_shape = (embed_dim, embed_dim)
            
            with torch.no_grad():
                # Adapt Q, K, V weights
                q_adapted = adapt_weight_tensor(pretrained_attn.query.weight, q_target_shape, "Query weight")
                k_adapted = adapt_weight_tensor(pretrained_attn.key.weight, k_target_shape, "Key weight")
                v_adapted = adapt_weight_tensor(pretrained_attn.value.weight, v_target_shape, "Value weight")
                
                # Copy adapted weights to the combined in_proj_weight tensor
                custom_attn.in_proj_weight[:embed_dim].copy_(q_adapted)
                custom_attn.in_proj_weight[embed_dim:2*embed_dim].copy_(k_adapted)
                custom_attn.in_proj_weight[2*embed_dim:].copy_(v_adapted)
                
                # Adapt and copy Q, K, V biases
                q_bias_adapted = adapt_bias_tensor(pretrained_attn.query.bias, (embed_dim,), "Query bias")
                k_bias_adapted = adapt_bias_tensor(pretrained_attn.key.bias, (embed_dim,), "Key bias")
                v_bias_adapted = adapt_bias_tensor(pretrained_attn.value.bias, (embed_dim,), "Value bias")
                
                custom_attn.in_proj_bias[:embed_dim].copy_(q_bias_adapted)
                custom_attn.in_proj_bias[embed_dim:2*embed_dim].copy_(k_bias_adapted)
                custom_attn.in_proj_bias[2*embed_dim:].copy_(v_bias_adapted)
                
                # Adapt and copy output projection
                out_weight_adapted = adapt_weight_tensor(
                    pretrained_layer.attention.output.dense.weight, 
                    custom_attn.out_proj.weight.shape, 
                    "Output projection weight"
                )
                out_bias_adapted = adapt_bias_tensor(
                    pretrained_layer.attention.output.dense.bias,
                    custom_attn.out_proj.bias.shape,
                    "Output projection bias"
                )
                
                custom_attn.out_proj.weight.copy_(out_weight_adapted)
                custom_attn.out_proj.bias.copy_(out_bias_adapted)
            
            # Adapt and copy layer norm weights
            with torch.no_grad():
                # First layer norm (before attention)
                ln1_weight_adapted = adapt_weight_tensor(
                    pretrained_layer.layernorm_before.weight,
                    custom_layer.norm1.weight.shape,
                    "LayerNorm1 weight"
                )
                ln1_bias_adapted = adapt_bias_tensor(
                    pretrained_layer.layernorm_before.bias,
                    custom_layer.norm1.bias.shape,
                    "LayerNorm1 bias"
                )
                
                custom_layer.norm1.weight.copy_(ln1_weight_adapted)
                custom_layer.norm1.bias.copy_(ln1_bias_adapted)
                
                # Second layer norm (after attention)
                ln2_weight_adapted = adapt_weight_tensor(
                    pretrained_layer.layernorm_after.weight,
                    custom_layer.norm2.weight.shape,
                    "LayerNorm2 weight"
                )
                ln2_bias_adapted = adapt_bias_tensor(
                    pretrained_layer.layernorm_after.bias,
                    custom_layer.norm2.bias.shape,
                    "LayerNorm2 bias"
                )
                
                custom_layer.norm2.weight.copy_(ln2_weight_adapted)
                custom_layer.norm2.bias.copy_(ln2_bias_adapted)
            
            # Adapt and copy feedforward weights
            pretrained_ff = pretrained_layer.intermediate
            pretrained_output = pretrained_layer.output
            custom_ff = custom_layer.linear1
            custom_output = custom_layer.linear2
            
            with torch.no_grad():
                # First feedforward layer
                ff1_weight_adapted = adapt_weight_tensor(
                    pretrained_ff.dense.weight,
                    custom_ff.weight.shape,
                    "Feedforward1 weight"
                )
                ff1_bias_adapted = adapt_bias_tensor(
                    pretrained_ff.dense.bias,
                    custom_ff.bias.shape,
                    "Feedforward1 bias"
                )
                
                custom_ff.weight.copy_(ff1_weight_adapted)
                custom_ff.bias.copy_(ff1_bias_adapted)
                
                # Second feedforward layer (output projection)
                ff2_weight_adapted = adapt_weight_tensor(
                    pretrained_output.dense.weight,
                    custom_output.weight.shape,
                    "Feedforward2 weight"
                )
                ff2_bias_adapted = adapt_bias_tensor(
                    pretrained_output.dense.bias,
                    custom_output.bias.shape,
                    "Feedforward2 bias"
                )
                
                custom_output.weight.copy_(ff2_weight_adapted)
                custom_output.bias.copy_(ff2_bias_adapted)
            
            print(f"    Layer {layer_idx} initialization completed successfully")
            
        except Exception as e:
            print(f"      Error adapting layer {layer_idx}: {e}")
            print(f"      Falling back to Xavier initialization for this layer")
            
            # Fallback: Initialize with Xavier uniform
            try:
                with torch.no_grad():
                    for param in custom_layer.parameters():
                        if len(param.shape) >= 2:
                            torch.nn.init.xavier_uniform_(param)
                        else:
                            torch.nn.init.zeros_(param)
                print(f"      Xavier initialization completed for layer {layer_idx}")
            except Exception as fallback_error:
                print(f"      Even fallback initialization failed: {fallback_error}")
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, seq_len) from conv layers
        Returns:
            Tensor of shape (batch_size, seq_len, final_output_dim)
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape: (batch_size, channels, seq_len) → (batch_size, seq_len, channels)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # Project to transformer dimension if needed
        if self.input_projection is not None:
            x = self.input_projection(x)  # (batch_size, seq_len, transformer_dim)
        
        # Reshape for transformer: (batch_size, seq_len, features) → (seq_len, batch_size, features)
        x = x.transpose(0, 1)  # (seq_len, batch_size, transformer_dim)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Apply transformer layers
        x = self.transformer_encoder(x)  # (seq_len, batch_size, transformer_dim)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Reshape back: (seq_len, batch_size, transformer_dim) → (batch_size, seq_len, transformer_dim)
        x = x.transpose(0, 1)
        
        return x