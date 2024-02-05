from typing import Union

import gpytorch as gpy
import torch

from .forest import AlfalfaForest, AlfalfaTree


class AlfalfaTreeModelKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree_model: Union[AlfalfaTree, AlfalfaForest]):
        super().__init__()
        self.tree_model = tree_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return torch.as_tensor(self.tree_model.gram_matrix(x1, x2)).float()

    def get_extra_state(self):
        return {"tree_model": self.tree_model.as_dict()}

    def set_extra_state(self, state):
        d = state["tree_model"]
        if d["tree_model_type"] == "tree":
            self.tree_model = AlfalfaTree.from_dict(d)
        elif d["tree_model_type"] == "forest":
            self.tree_model = AlfalfaForest.from_dict(d)


class AlfalfaGP(gpy.models.ExactGP):
    def __init__(
        self, train_inputs, train_targets, likelihood, tree_model: AlfalfaTree
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        tree_kernel = AlfalfaTreeModelKernel(tree_model)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def tree_model(self) -> Union[AlfalfaTree, AlfalfaForest]:
        return self.covar_module.base_kernel.tree_model

    @classmethod
    def from_mcmc_samples(cls, model: "AlfalfaGP", samples):
        likelihood = gpy.likelihoods.GaussianLikelihood()
        all_trees = {"tree_model_type": "forest", "trees": []}
        for sample in samples:
            forest_dict = sample["covar_module.base_kernel._extra_state"]["tree_model"]
            all_trees["trees"] += forest_dict["trees"]

        tree_model = AlfalfaForest.from_dict(all_trees)
        tree_model.initialise(model.tree_model.space)
        gp = cls(model.train_inputs[0], model.train_targets, likelihood, tree_model)

        avg_noise = torch.mean(
            torch.tensor([s["likelihood.noise_covar.raw_noise"] for s in samples])
        )
        likelihood.noise = torch.nn.Softplus(avg_noise)

        avg_scale = torch.mean(
            torch.tensor([s["covar_module.raw_outputscale"] for s in samples])
        )
        gp.covar_module.outputscale = torch.nn.Softplus(avg_scale)

        return gp
