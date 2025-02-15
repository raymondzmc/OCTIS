"""Class for loading BOW dataset."""

import torch
from torch.utils.data import Dataset
from octis.optimization.coherence_loss import compute_pairwise_coherence_matrix


class BOWDataset(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, X, idx2token):
        """
        Initialize NewsGroupDataset.

        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        self.X = X
        self.idx2token = idx2token
        self.pairwise_coherence = None
            

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        X = torch.FloatTensor(self.X[i])

        return {'X': X}
    
    def compute_coherence_weight(self, corpus, idx2token):
        """Compute the pairwise coherence matrix used in coherence loss."""
        self.coherence_weight = compute_pairwise_coherence_matrix(corpus=corpus, idx2token=idx2token)

