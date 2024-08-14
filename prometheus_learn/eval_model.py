import torch
import torch.nn as nn
import numpy as np
from util import get_mirror


class EvalModel(nn.Module):
    """Evaluation of the sparse input bitboards.

    Args:
        weights_initial (np.array): array holding all weights
        (likely loaded from util.weights_from_file).
    """

    def __init__(self, weights_initial: torch.Tensor):
        super(EvalModel, self).__init__()
        self.weights = nn.Parameter(weights_initial)
        self.register_buffer("mirror", torch.tensor(get_mirror()))

    def forward(self, white_indices: torch.Tensor, black_indices: torch.Tensor):
        return torch.sum(
            torch.cat(
                self.weights[white_indices], -self.weights[self.mirror[black_indices]]
            )
        )
