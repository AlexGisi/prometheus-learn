import torch
import torch.nn as nn
import numpy as np


class EvalModel(nn.Module):
    """Evaluation of the sparse input bitboards.

    Args:
        weights_initial (np.array): array holding all weights 
        (likely loaded from util.weights_from_file).
    """
    def __init__(
        self, 
        weights_initial: np.array
    ):
        super(EvalModel, self).__init__()
        self.weights = nn.Parameter(torch.from_numpy(weights_initial))
        
    def forward(self, non_zero_indices):
        return torch.sum(self.weights[non_zero_indices])
