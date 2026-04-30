import torch
import torch.nn as nn
import math

class GenotypeModelWithTransformer(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, act, dropout_rate, num_covariates=10, 
                 num_transformer_layers=3, d_model=256, nhead=8, fc_layers=[256], print_dimensions=False):
        super(GenotypeModelWithTransformer, self).__init__()
        self.print_dimensions = print_dimensions
        self.has_printed_dimensions = False

        self.input_channels = 3

        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")

        # Max pooling layer
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Project convolutional output to d_model dimensions
        self.proj = nn.Linear(conv_channels[-1], d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)

        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Fully connected layers
        fc_input_size = d_model * 3 + num_covariates  # *3 because we concatenate max_pool, cls_token, and avg_pool
        self.fc_layers = self._create_fc_layers(fc_input_size, fc_layers, dropout_rate, act)

        # Final classification layer
        self.classifier = nn.Sequential(
            self.fc_layers,
            nn.Linear(fc_layers[-1], 1),
            nn.Sigmoid()
        )

    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        layers = []
        for i in range(len(conv_channels)):
            layers.append(nn.Conv1d(in_channels=self.input_channels if i == 0 else conv_channels[i-1],
                                    out_channels=conv_channels[i],  
                                    kernel_size=kernel_sizes[i],
                                    stride=stride[i]))
            layers.append(nn.BatchNorm1d(conv_channels[i]))
            layers.append(self.get_activation(act))
            if i > 0:  # Apply dropout after the first layer
                layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _create_fc_layers(self, input_size, fc_sizes, dropout_rate, act):
        layers = []
        for i, fc_size in enumerate(fc_sizes):
            layers.append(nn.Linear(input_size if i == 0 else fc_sizes[i-1], fc_size))
            layers.append(self.get_activation(act))
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 3, input_size, dtype=torch.float32)
        x = self.conv_layers(x)
        return x.shape[2]  # Return the sequence length after convolutions

    def forward(self, x, covariates):
        if self.print_dimensions and not self.has_printed_dimensions:
            print("\nPrinting model dimensions for the first batch:")
            print(f"Input shape: {x.shape}")
        
        x = self.conv_layers(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After conv layers: {x.shape}")
        
        # Apply max pooling
        max_pooled = self.max_pool(x).squeeze(2)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After max pooling: {max_pooled.shape}")
        
        # Convert (B, C, L) to (L, B, C)
        x = x.permute(2, 0, 1)
        
        # Project to d_model dimensions
        x = self.proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((cls_tokens, x), dim=0)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        x = self.transformer(x)
        
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After transformer layers: {x.shape}")
        
        # Extract CLS token representation
        cls_representation = x[0]
        
        # Apply average pooling
        avg_pooled = x[1:].mean(dim=0)
        
        # Concatenate max_pooled, cls_representation, and avg_pooled
        combined = torch.cat([max_pooled, cls_representation, avg_pooled], dim=1)
        
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"Combined pooled representation: {combined.shape}")
            print(f"Covariates shape: {covariates.shape}")
        
        # Concatenate with covariates
        combined = torch.cat([combined, covariates], dim=1)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After concatenating covariates: {combined.shape}")
        
        x = self.classifier(combined)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"Final output: {x.shape}")
            self.has_printed_dimensions = True
        
        return x.squeeze(1)
    
    def get_activation(self, name):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)