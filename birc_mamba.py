import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

try:
    from mamba_ssm.modules.mamba_simple import Mamba, Block
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.block import Block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out


class BiRCMamba(nn.Module):
    """
    Bidirectional Reverse Complement Mamba for DNA sequences
    This module handles both bidirectional processing and reverse complement properties of DNA
    """

    def __init__(
            self,
            d_model=128,
            n_layer=3,
            d_state=16,
            expand_factor=2,
            d_conv=4,
            conv_bias=True,
            dropout=0.1,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            bias=False,
            bidirectional_strategy="add"
    ):
        super().__init__()
        self.d_model = d_model

        # DNA embedding - from one-hot to model dimension
        self.embedding = nn.Linear(4, d_model)

        # Layers that process original DNA sequence and its reverse complement
        self.mamba_layers = nn.ModuleList([
            BiMambaWrapper(
                d_model=d_model // 2,  # Half for original, half for reverse complement
                bidirectional=True,
                bidirectional_strategy=bidirectional_strategy,
                d_state=d_state,
                expand=expand_factor,
                d_conv=d_conv,
                conv_bias=conv_bias,
                bias=bias,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
            )
            for _ in range(n_layer)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)

        # Final MLP
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def _get_reverse_complement(self, x):
        """
        Get reverse complement of DNA sequence in one-hot encoding

        Args:
            x: One-hot encoded DNA (batch_size, seq_len, 4)

        Returns:
            Reverse complement in one-hot encoding
        """
        # Reverse the sequence
        x_rev = torch.flip(x, dims=[1])

        # Complement the bases (A<->T, G<->C)
        # A[0] <-> T[1], C[2] <-> G[3]
        # Apply mapping to get complementary bases
        # We can simply swap the columns
        return x_rev[:, :, torch.tensor([1, 0, 3, 2])]

    def forward(self, x):
        """
        Forward pass through BiRC-Mamba

        Args:
            x: One-hot encoded DNA sequence (batch_size, seq_len, 4)

        Returns:
            Processed features
        """
        batch_size, seq_len, _ = x.shape

        # Get reverse complement
        x_rc = self._get_reverse_complement(x)

        # Embed original sequence and reverse complement
        x_embed = self.embedding(x)  # (batch_size, seq_len, d_model)
        x_rc_embed = self.embedding(x_rc)  # (batch_size, seq_len, d_model)

        # Process through Mamba layers
        for layer in self.mamba_layers:
            # Split channels for original and reverse complement processing
            x_orig, x_comp = torch.chunk(x_embed, 2, dim=-1)
            x_rc_orig, x_rc_comp = torch.chunk(x_rc_embed, 2, dim=-1)

            # Process original sequence
            y_orig = layer(x_orig)

            # Process reverse complement
            y_rc = layer(x_rc_orig)
            y_rc_flipped = torch.flip(y_rc, dims=[1])  # Flip back to align with original

            # Combine results according to the formula:
            # M_RC,θ(X) = concat([M_θ(X^(1:H/2)), RC(M_θ(RC(X^(H/2:H))))])
            x_embed = torch.cat([y_orig, y_rc_flipped], dim=-1)

            # Also update the reverse complement embedding for the next layer
            x_rc_embed = torch.cat([y_rc, torch.flip(y_orig, dims=[1])], dim=-1)

        # Apply layer normalization
        x_norm = self.layer_norm(x_embed)

        # Apply self-attention
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)

        # Final MLP
        x_fc1 = self.relu(self.fc1(attn_output))
        x_fc1 = self.dropout(x_fc1)
        x_out = self.fc2(x_fc1)

        return x_out


# Function to convert DNA sequence to one-hot encoding
def dna_to_onehot(sequence):
    """
    Convert DNA sequence to one-hot encoding

    Args:
        sequence: String of DNA sequence (ATCG)

    Returns:
        One-hot encoded tensor of shape (1, seq_length, 4)
    """
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # Convert to indices
    indices = [mapping.get(base, 0) for base in sequence.upper()]

    # Convert to one-hot
    onehot = F.one_hot(torch.tensor(indices), num_classes=4).float()

    # Add batch dimension
    return onehot.unsqueeze(0)  # shape: (1, seq_length, 4)

