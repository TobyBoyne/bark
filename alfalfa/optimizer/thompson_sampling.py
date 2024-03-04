"""Generate samples of f^*

This uses a similar formulation to the UCB. Here, however, we sample posterior
draws from the GP, then take the minimum."""

import torch

from ..tree_kernels import AlfalfaGP


def generate_fstar_samples(
    model: AlfalfaGP, num_samples: int = 10, maximise: bool = False
):
    """Generate samples of function optima."""

    sample_sites = model.train_inputs[0]
    i = torch.zeros((sample_sites.shape[0], 1), dtype=torch.long)
    model.eval()
    function_values = model(sample_sites, i).sample(torch.Size([num_samples]))
    if maximise:
        f_star = function_values.max(dim=0).values
    else:
        f_star = function_values.min(dim=0).values

    return f_star
