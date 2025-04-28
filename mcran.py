import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


coden_dict1 = {'GCT': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
               'TGT': 1, 'TGC': 1,  # systeine<C>
               'GAT': 2, 'GAC': 2,  # aspartic acid<D>
               'GAA': 3, 'GAG': 3,  # glutamic acid<E>
               'TTT': 4, 'TTC': 4,  # phenylanaline<F>
               'GGT': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
               'CAT': 6, 'CAC': 6,  # histidine<H>
               'ATT': 7, 'ATC': 7, 'ATA': 7,  # isoleucine<I>
               'AAA': 8, 'AAG': 8,  # lycine<K>
               'TTA': 9, 'TTG': 9, 'CTT': 9, 'CTC': 9, 'CTA': 9, 'CTG': 9,  # leucine<L>
               'ATG': 10,  # methionine<M>
               'AAT': 11, 'AAC': 11,  # asparagine<N>
               'CCT': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
               'CAA': 13, 'CAG': 13,  # glutamine<Q>
               'CGT': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
               'TCT': 15, 'TCC': 15, 'TCA': 15, 'TCG': 15, 'AGT': 15, 'AGC': 15,  # serine<S>
               'ACT': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
               'GTT': 17, 'GTC': 17, 'GTA': 17, 'GTG': 17,  # valine<V>
               'TGG': 18,  # tryptophan<W>
               'TAT': 19, 'TAC': 19,  # tyrosine(Y)
               'TAA': 20, 'TAG': 20, 'TGA': 20,  # STOP code
               }


def coden1(seq):
    """
    Convert DNA sequence to SCE encoding
    """
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq) - 2):
        codon = seq[i:i + 3]
        if codon in coden_dict1:
            vectors[i][coden_dict1[codon]] = 1
    return vectors.tolist()  # Convert matrix to list


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """

    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Ensure input_dim is divisible by num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # Define query, key, value projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output linear layer
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()

        # Apply linear projections
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back to input dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, input_dim)
        output = self.out_proj(attn_output)

        return output


class MCRAN(nn.Module):
    """
    Multi-scale Convolutional Recurrent Attention Network module
    for DNA-protein binding site prediction
    """

    def __init__(self, seq_length=101, dropout_rate=0.3):
        super(MCRAN, self).__init__()

        # Input dimensions
        self.input_dim = 21  # From SCE encoding
        self.seq_length = seq_length

        # First level multi-scale CNN
        # Branch 1: kernel_size=3
        self.conv1_branch1 = nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1)
        self.bn1_branch1 = nn.BatchNorm1d(64)

        # Branch 2: kernel_size=5
        self.conv1_branch2 = nn.Conv1d(self.input_dim, 64, kernel_size=5, padding=2)
        self.bn1_branch2 = nn.BatchNorm1d(64)

        # Branch 3: kernel_size=7
        self.conv1_branch3 = nn.Conv1d(self.input_dim, 64, kernel_size=7, padding=3)
        self.bn1_branch3 = nn.BatchNorm1d(64)

        # Projection for residual connection
        self.conv_proj = nn.Conv1d(self.input_dim, 192, kernel_size=1)

        # Second level CNN
        self.conv2_1 = nn.Conv1d(192, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.conv2_proj = nn.Conv1d(192, 128, kernel_size=1)  # For residual connection

        # BiLSTM layers
        self.lstm1 = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = MultiHeadSelfAttention(128)

        # Final fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the MCRAN module

        Args:
            x: Input tensor of shape (batch_size, seq_length, 21) from SCE encoding

        Returns:
            Probability of DNA-protein binding
        """
        # Transpose for CNN (batch_size, seq_length, input_dim) -> (batch_size, input_dim, seq_length)
        x_cnn = x.transpose(1, 2)

        # First level multi-scale CNN
        # Branch 1
        branch1 = F.relu(self.bn1_branch1(self.conv1_branch1(x_cnn)))
        branch1 = F.max_pool1d(branch1, kernel_size=2, stride=1, padding=1)

        # Branch 2
        branch2 = F.relu(self.bn1_branch2(self.conv1_branch2(x_cnn)))
        branch2 = F.max_pool1d(branch2, kernel_size=2, stride=1, padding=1)

        # Branch 3
        branch3 = F.relu(self.bn1_branch3(self.conv1_branch3(x_cnn)))
        branch3 = F.max_pool1d(branch3, kernel_size=2, stride=1, padding=1)

        # Concatenate branches
        concat_features = torch.cat([branch1, branch2, branch3], dim=1)  # (batch_size, 192, seq_length)

        # Residual connection
        res1 = concat_features + self.conv_proj(x_cnn)
        res1 = self.dropout(res1)

        # Second level CNN
        conv2_1_out = F.relu(self.bn2_1(self.conv2_1(res1)))
        conv2_2_out = F.relu(self.bn2_2(self.conv2_2(conv2_1_out)))

        # Residual connection in second level
        res2 = conv2_2_out + self.conv2_proj(res1)
        res2 = F.max_pool1d(res2, kernel_size=2, stride=1, padding=1)
        res2 = self.dropout(res2)

        # Transpose back for LSTM (batch_size, channels, seq_length) -> (batch_size, seq_length, channels)
        x_lstm = res2.transpose(1, 2)

        # First BiLSTM layer
        lstm1_out, _ = self.lstm1(x_lstm)
        lstm1_out = self.recurrent_dropout(lstm1_out)

        # Second BiLSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.recurrent_dropout(lstm2_out)

        # Multi-head self-attention
        attn_out = self.attention(lstm2_out)

        # Global pooling
        global_avg_pool = torch.mean(attn_out, dim=1)  # (batch_size, 128)
        global_max_pool, _ = torch.max(attn_out, dim=1)  # (batch_size, 128)

        # Concatenate global pools
        concat_pool = torch.cat([global_avg_pool, global_max_pool], dim=1)  # (batch_size, 256)

        # Fully connected layers
        fc1_out = F.relu(self.bn_fc1(self.fc1(concat_pool)))
        fc1_out = self.dropout(fc1_out)

        fc2_out = F.relu(self.bn_fc2(self.fc2(fc1_out)))
        fc2_out = self.dropout(fc2_out)

        # Final layer with sigmoid activation
        output = torch.sigmoid(self.fc3(fc2_out))

        return output


def preprocess_dna_for_mcran(dna_sequence):
    """
    Convert DNA sequence to SCE encoding and prepare for MCRAN input

    Args:
        dna_sequence: String DNA sequence of length 101

    Returns:
        Tensor of shape (1, 101, 21) for MCRAN input
    """
    # Apply SCE encoding
    sce_encoded = coden1(dna_sequence)

    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(sce_encoded).unsqueeze(0)  # shape: (1, seq_length, 21)

    return tensor



