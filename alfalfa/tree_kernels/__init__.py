import gpytorch

from .multitask_likelihood import MultitaskGaussianLikelihood
from .tree_gps import AlfalfaGP, AlfalfaMOGP, AlfalfaSampledModel
from .tree_model_kernel import AlfalfaTreeModelKernel

AnyModel = gpytorch.models.ExactGP | AlfalfaSampledModel
