import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNetDBP(nn.Module):

    """
    Multi-modal Deep Network for DNA-Binding Protein prediction
    Combines MCRAN and BiRC-Mamba with MFFS
    """

    def __init__(self, seq_length=101, mcran_dim=128, bircmamba_dim=128, fusion_dim=128, dropout_rate=0.3):
        super(MDNetDBP, self).__init__()

        # MCRAN module (from the provided code)
        self.mcran = MCRAN(seq_length=seq_length, dropout_rate=dropout_rate)

        # BiRC-Mamba module (from the provided code)
        self.bircmamba = BiRCMamba(d_model=bircmamba_dim, n_layer=3, d_state=16, dropout=dropout_rate)

        # Feature extractors to get intermediate features
        self.mcran_feature_extractor = nn.Linear(256, mcran_dim)  # Assuming 256 is from concatenated global pools

        # Multi-level Feature Fusion Strategy
        self.mffs = MFFS(
            mcran_dim=mcran_dim,
            bircmamba_dim=bircmamba_dim,
            fusion_dim=fusion_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, dna_seq_sce, dna_seq_onehot):
        """
        Forward pass through the MDNet-DBP

        Args:
            dna_seq_sce: DNA sequence encoded with SCE for MCRAN
            dna_seq_onehot: DNA sequence one-hot encoded for BiRC-Mamba

        Returns:
            Final prediction probability
        """
        # Extract features and prediction from MCRAN
        # Here we need a modified version of MCRAN that returns intermediate features
        mcran_output = self.mcran(dna_seq_sce)

        # Since the original MCRAN doesn't expose intermediate features,
        # we assume we can extract the features before the final sigmoid
        # In practice, the MCRAN class would need to be modified to return these features

        # Extract features and prediction from BiRC-Mamba
        bircmamba_features = self.bircmamba(dna_seq_onehot)

        # Extract global features
        # Assuming bircmamba_features shape is [batch_size, seq_len, d_model]
        # We need to get a global representation
        global_avg_pool = torch.mean(bircmamba_features, dim=1)  # [batch_size, d_model]
        global_max_pool, _ = torch.max(bircmamba_features, dim=1)  # [batch_size, d_model]
        bircmamba_global_features = torch.cat([global_avg_pool, global_max_pool], dim=1)

        # Use feature extractors to get to expected dimensions
        mcran_features = self.mcran_feature_extractor(mcran_output)

        # Apply MFFS
        final_prediction, individual_predictions = self.mffs(mcran_features, bircmamba_global_features)

        return final_prediction, individual_predictions


def preprocess_dna_for_mdnetdbp(dna_sequence):
    """
    Convert DNA sequence to both SCE encoding and one-hot encoding

    Args:
        dna_sequence: String DNA sequence of length 101

    Returns:
        Tuple of tensors for MCRAN and BiRC-Mamba inputs
    """
    # SCE encoding for MCRAN
    sce_encoded = coden1(dna_sequence)
    sce_tensor = torch.FloatTensor(sce_encoded).unsqueeze(0)  # shape: (1, seq_length, 21)

    # One-hot encoding for BiRC-Mamba
    onehot_tensor = dna_to_onehot(dna_sequence)  # shape: (1, seq_length, 4)

    return sce_tensor, onehot_tensor