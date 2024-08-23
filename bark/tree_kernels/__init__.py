import gpytorch

from .multitask_likelihood import MultitaskGaussianLikelihood
from .tree_gps import BARKMOGP, BARKMixtureModel, LeafGP
from .tree_model_kernel import BARKTreeModelKernel

AnyModel = gpytorch.models.ExactGP | BARKMixtureModel
