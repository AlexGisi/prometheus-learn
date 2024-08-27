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
        assert weights_initial.size() == torch.Size([384])

        super(EvalModel, self).__init__()
        self.weights = nn.Parameter(weights_initial)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): weight and black position indicators concatenated, i.e. a 768x1 vector.
            The black tensor should be mirrored.
        """
        weights_full = torch.div(torch.cat([self.weights, -self.weights], dim=0), 100)  # (,768)
        pawn_evaluation = x @ weights_full  # inner product
        eval = torch.tanh(pawn_evaluation)
        return eval
