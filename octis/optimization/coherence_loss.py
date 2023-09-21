import math
import torch
import torch.nn as nn
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
from typing import Union

import warnings

"""
For more details on the notation used in `config`, please refer to our paper "Diversity-Aware Coherence Loss for Improving Neural Topic Models":
https://aclanthology.org/2023.acl-short.145.pdf
"""

# You can modify the hyperparameters configuration of the loss here
default_config = {
    "Mc_n": 20, # Top-n words of each topic for constructing the coherence mask (M_c) for weighting the pairwise coherence matric
    "Md_n": 20, # Top-n words of each topic for constructing the diversity mask (M_d) for weighting high and low-probability words
    "lambda_d": 0.7, # Balancing constant for weighting the loss between high-probability words and low-probability words
    "lambda_a": 100, # The weight applied to balance the magnitude between ELBO loss and Diversity-Aware Coherence (DAC) Loss
    "warmup_ratio": 0.5 # Ratio of warmup steps for linearly increasing lambda_a (TODO: consider other scheduler)
}

def compute_pairwise_coherence_matrix(corpus: list[list[str]], idx2token: dict[int, str], metric: str ='npmi'):
    """
    Compute the pairwise coherence scores for all vocab words.
    
    Parameters
    ----------
    corpus:
        Tokenized corpus, needed for coherence models that use sliding window based 
        (i.e. coherence=`c_npmi`) probability estimator.
    idx2token:
        Used by the Gensim's CoherenceModel as indices to compute the pairwise coherence
        matrix, NEEDS to be aligned with the model's output dimension.
    metric: optional, default: 'npmi'
        The coherence metric used to compute the pairwise matrix, default to 'npmi'
        as proposed by the original authors.
    """
    assert set([tok for doc in corpus for tok in doc]) == set(idx2token.values()), \
           "Mismatch between tokens extracted from the corpus and the idx2token mapping!"

    vocabulary = list(idx2token.values())
    vocab_size = len(vocabulary)

    dictionary = Dictionary()
    dictionary.id2token = idx2token
    dictionary.token2id = {v:k for k, v in idx2token.items()}
    pairwise_coherence_matrix = np.zeros((vocab_size, vocab_size))

    if metric == 'npmi':
        # Since Gensim only provides a single coherence score, there's no high-level API for computing pairwise coherence
        cm = CoherenceModel(topics=[vocabulary], texts=corpus, dictionary=dictionary, coherence='c_npmi', topn=vocab_size, processes=1)
        segmented_topics = cm.measure.seg(cm.topics)
        print("Accumulate word co-occurrences from corpus, this operation may take quite some time.")
        accumulator = cm.estimate_probabilities(segmented_topics)
        num_docs = accumulator.num_docs
        eps = 1e-12
        for w1, w2 in tqdm(segmented_topics[0], desc=f"Finished accumulation, computing pairwise `{metric}` matrix."):
            w1_count = accumulator[w1]
            w2_count = accumulator[w2]
            co_occur_count = accumulator[w1, w2]
            p_w1_w2 = co_occur_count / num_docs
            p_w1 = (w1_count / num_docs)
            p_w2 = (w2_count / num_docs)
            pairwise_coherence_matrix[w1, w2] = np.log((p_w1_w2 + eps) / (p_w1 * p_w2)) / -np.log(p_w1_w2  + eps)
    else:
        raise Exception(f"Coherence metric `{metric}` is not supported for pairwise matrix used in the Diversity-Aware Coherence loss.")
    return pairwise_coherence_matrix




class CoherenceLoss(nn.Module):
    """
    Implementation of the Diversity-Aware Coherence loss as described in Sec. 3 of paper
    "Diversity-Aware Coherence Loss for Improving Neural Topic Models".
    For more details on the notation used in `config`, please refer to our paper:
    https://aclanthology.org/2023.acl-short.145.pdf
    """
    def __init__(self,
                 coherence_weight: np.array,
                 num_topics: int,
                 num_epochs: int,
                 config: dict[str, Union[float, int]] = default_config):

        super(CoherenceLoss, self).__init__()
        self.coherence_weight = torch.Tensor(coherence_weight)
        self.num_topics = num_topics
        self.num_epochs = num_epochs
        self.config = config
        assert self.config['lambda_d'] >= 0.5 and self.config['lambda_d'] <= 1, \
            "Hyperparameter `lambda_d` must be between 0.5 and 1."
        self.coherence_weight.fill_diagonal_(1)
    
    def normalize(self, matrix):
        """
        In-place row-wise normalization
        """
        for row_idx, row in enumerate(matrix):
            row_min = row.min().item()
            row_max = row.max().item()
            matrix[row_idx] = (row - row_min)/(row_max - row_min)
        return matrix
    
    def forward(self, beta, epoch):

        # Move coherence_weight to same device as beta
        if beta.device != self.coherence_weight.device:
            self.coherence_weight = self.coherence_weight.to(beta.device)

        Mc_index = torch.topk(beta, self.config['Mc_n'], dim=1)[1]
        Mc = torch.zeros_like(beta)
        for row_idx, indices in enumerate(Mc_index):
            Mc[row_idx, indices] = 1
        Mc = (1 - Mc) * -99999

        Wc = 1 - self.normalize(torch.matmul(torch.softmax(beta + Mc, dim=1).detach(),
                                             self.coherence_weight))
        
        # We find that further increasing the magnitude by 100 helps improve performance
        loss = 100 * (torch.softmax(beta, dim=1) ** 2) * Wc

        if self.config['lambda_d'] > 0.5:
            Md_index = torch.topk(beta, self.config['Md_n'], dim=1)[1]
            Md_mask = torch.zeros_like(beta)
            for row_idx, indices in enumerate(Md_index):
                Md_mask[row_idx, indices] = 1
            Md_mask = Md_mask.bool()
            Md = torch.zeros_like(beta).bool()
            for topic_idx in range(self.num_topics):
                other_rows_mask = torch.ones(self.num_topics).bool().to(beta.device)
                other_rows_mask[topic_idx] = False
                Md[topic_idx] = Md_mask[other_rows_mask].sum(0) > 0
            
            loss = (torch.masked_select(loss, Md)).sum() * self.config['lambda_d']  + \
                (torch.masked_select(loss, ~Md)).sum() * (1 - self.config['lambda_d'])
            loss *= 2
        
        warmup_epochs = math.floor(self.config['warmup_ratio'] * self.num_epochs)
        lambda_a_delta = self.config["lambda_a"] / warmup_epochs

        if epoch < warmup_epochs:
            lambda_a = epoch * lambda_a_delta
        else:
            lambda_a = self.config["lambda_a"]

        return lambda_a * loss.sum()