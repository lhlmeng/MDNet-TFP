import numpy as np
import torch


# Convert string labels to binary format (0 if all zeros, 1 if any ones)
labels = []
for label_str in labels_raw:
    if '1' in label_str:
        labels.append(1)
    else:
        labels.append(0)

print(f"Binary labels: {labels}")

# Create the codon dictionary for DNA
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


# Function for SCE encoding
def coden1(seq):
    """Convert DNA sequence to SCE encoding"""
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq) - 2):
        codon = seq[i:i + 3]
        if codon in coden_dict1:
            vectors[i][coden_dict1[codon]] = 1
    return vectors


# Function for one-hot encoding
def one_hot_encode(seq):
    """Convert DNA sequence to one-hot encoding"""
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]}

    one_hot = np.zeros((len(seq), 4))
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i] = mapping[nucleotide]
    return one_hot


# Function to get reverse complement
def reverse_complement(seq):
    """Get reverse complement of DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


# Process all sequences
processed_data = []
for i, seq in enumerate(sequences):
    # 1. SCE encoding for MCRAN module
    sce_encoded = coden1(seq)

    # 2. One-hot encoding for BiRC-Mamba module
    one_hot_encoded = one_hot_encode(seq)

    # 3. Get reverse complement and its one-hot encoding
    rev_comp_seq = reverse_complement(seq)
    rev_comp_one_hot = one_hot_encode(rev_comp_seq)

    # Store processed data
    processed_data.append({
        'sequence': seq,
        'label': labels[i],
        'sce_encoding': sce_encoded,
        'one_hot_encoding': one_hot_encoded,
        'reverse_complement': rev_comp_seq,
        'rev_comp_one_hot': rev_comp_one_hot
    })

# Convert to PyTorch tensors for the model
for item in processed_data:
    item['sce_tensor'] = torch.FloatTensor(item['sce_encoding']).unsqueeze(0)  # Add batch dimension
    item['one_hot_tensor'] = torch.FloatTensor(item['one_hot_encoding']).unsqueeze(0)  # Add batch dimension
    item['rev_comp_tensor'] = torch.FloatTensor(item['rev_comp_one_hot']).unsqueeze(0)  # Add batch dimension

# Print summary of the first processed sequence
print(f"\nOriginal sequence: {processed_data[0]['sequence']}")
print(f"Binary label: {processed_data[0]['label']}")
print(f"SCE encoding shape: {processed_data[0]['sce_tensor'].shape}")
print(f"One-hot encoding shape: {processed_data[0]['one_hot_tensor'].shape}")
print(f"Reverse complement: {processed_data[0]['reverse_complement']}")
print(f"Reverse complement one-hot shape: {processed_data[0]['rev_comp_tensor'].shape}")
