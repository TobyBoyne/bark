import gpytorch as gpy
import torch
from beartype.typing import Optional

from ..forest import AlfalfaForest


class AlfalfaTreeModelKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree_model: Optional[AlfalfaForest]):
        super().__init__()
        self.tree_model = tree_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return torch.as_tensor(
            self.tree_model.gram_matrix(x1.detach().numpy(), x2.detach().numpy())
        )

    def get_extra_state(self):
        return {"tree_model": self.tree_model.as_dict()}

    def set_extra_state(self, state):
        d = state["tree_model"]
        self.tree_model = AlfalfaForest.from_dict(d)
