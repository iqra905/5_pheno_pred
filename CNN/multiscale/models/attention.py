# multiscale/models/attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseSpecificAttention(nn.Module):
    """Disease-specific attention mechanism"""
    
    def __init__(self, feature_dim, num_diseases, num_heads=8):
        super(DiseaseSpecificAttention, self).__init__()
        self.num_diseases = num_diseases
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        # Disease-specific query generators
        self.disease_queries = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_diseases)
        ])
        
        # Shared key and value projections
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.size()
        
        # Generate keys and values (shared across diseases)
        keys = self.key_proj(x)  # (batch_size, seq_len, feature_dim)
        values = self.value_proj(x)  # (batch_size, seq_len, feature_dim)
        
        # Reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        disease_outputs = []
        
        for disease_idx in range(self.num_diseases):
            # Generate disease-specific queries
            # Use global average pooling to create a single query vector per disease
            global_context = torch.mean(x, dim=1)  # (batch_size, feature_dim)
            queries = self.disease_queries[disease_idx](global_context)  # (batch_size, feature_dim)
            queries = queries.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            # Reshape queries for multi-head attention
            queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            # queries shape: (batch_size, num_heads, 1, head_dim)
            
            # Compute attention scores
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores shape: (batch_size, num_heads, 1, seq_len)
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attended = torch.matmul(attention_weights, values)
            # attended shape: (batch_size, num_heads, 1, head_dim)
            
            # Reshape and project
            attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, feature_dim)
            attended = self.output_proj(attended)
            
            # Squeeze to get (batch_size, feature_dim)
            disease_outputs.append(attended.squeeze(1))
        
        # Stack disease-specific features
        return torch.stack(disease_outputs, dim=1)  # (batch_size, num_diseases, feature_dim)


class SeparateDiseaseHead(nn.Module):
    """Separate prediction head for each disease"""
    
    def __init__(self, input_dim, hidden_dims, dropout_rate, act):
        super(SeparateDiseaseHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(act),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def _get_activation(self, name):
        """Get activation function by name"""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        return self.head(x)