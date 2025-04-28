import torch
import torch.nn as nn
import torch.nn.functional as F


class MFFS(nn.Module):
    """
    Multi-level Feature Fusion Strategy module for combining features
    from MCRAN and BiRC-Mamba modules
    """

    def __init__(self, mcran_dim=128, bircmamba_dim=128, fusion_dim=128, dropout_rate=0.3):
        super(MFFS, self).__init__()

        # Dimensions
        self.mcran_dim = mcran_dim
        self.bircmamba_dim = bircmamba_dim
        self.fusion_dim = fusion_dim

        # Feature-level fusion components

        # Feature alignment layers
        self.mcran_align = nn.Linear(mcran_dim, fusion_dim)
        self.bircmamba_align = nn.Linear(bircmamba_dim, fusion_dim)

        # Adaptive feature fusion
        self.adaptive_weight = nn.Sequential(
            nn.Linear(fusion_dim * 2, 1),
            nn.Sigmoid()
        )

        # Decision-level fusion components

        # Individual prediction heads
        self.mcran_pred = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.bircmamba_pred = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.fused_pred = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Gating mechanism for decision fusion
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, mcran_features, bircmamba_features):
        """
        Forward pass through the MFFS module

        Args:
            mcran_features: Features from MCRAN module
            bircmamba_features: Features from BiRC-Mamba module

        Returns:
            Final prediction probability and individual predictions
        """
        batch_size = mcran_features.size(0)

        # Feature-level fusion

        # 1. Feature alignment
        mcran_aligned = self.mcran_align(mcran_features)
        bircmamba_aligned = self.bircmamba_align(bircmamba_features)

        # 2. Adaptive feature fusion
        # Calculate adaptive weight alpha
        concat_features = torch.cat([mcran_aligned, bircmamba_aligned], dim=1)
        alpha = self.adaptive_weight(concat_features)

        # Fuse features using learned weight
        fused_features = alpha * mcran_aligned + (1 - alpha) * bircmamba_aligned

        # Apply dropout for regularization
        fused_features = self.dropout(fused_features)

        # Decision-level fusion

        # 1. Get individual predictions
        p1 = self.mcran_pred(mcran_aligned)  # MCRAN prediction
        p2 = self.bircmamba_pred(bircmamba_aligned)  # BiRC-Mamba prediction
        p_fused = self.fused_pred(fused_features)  # Fused features prediction

        # 2. Calculate gating weights for decision fusion
        # Concatenate all features for gate input
        gate_input = torch.cat([mcran_aligned, bircmamba_aligned, fused_features], dim=1)
        weights = self.gate(gate_input)  # [w1, w2, w_fused]

        # 3. Final weighted prediction
        p_final = weights[:, 0].unsqueeze(1) * p1 + \
                  weights[:, 1].unsqueeze(1) * p2 + \
                  weights[:, 2].unsqueeze(1) * p_fused

        return p_final, {'mcran': p1, 'bircmamba': p2, 'fused': p_fused, 'weights': weights}