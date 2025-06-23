import warnings

import torch
from jaxtyping import Float

from ..tree_kernels import BARKGP


def generate_fstar_samples(
    model: BARKGP, num_samples: int = 10, maximise: bool = False
) -> Float[torch.Tensor, "{num_samples}"]:
    """Generate samples of function optima."""

    sample_sites = model.train_inputs[0]
    i = torch.zeros((sample_sites.shape[0], 1), dtype=torch.long)
    model.eval()
    with warnings.catch_warnings():
        # sampling from GP warns due to not being pd
        # which is expected for a tree kernel
        warnings.simplefilter("ignore")
        function_values = model(sample_sites, i).sample(torch.Size([num_samples]))
    if maximise:
        f_star = function_values.max(dim=-1).values
    else:
        f_star = function_values.min(dim=-1).values

    return f_star
