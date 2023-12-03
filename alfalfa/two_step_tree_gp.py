import gpytorch as gpy
import lightgbm as lgb
from leaf_gp.gbm_model import GbmModel
from leaf_gp.lgbm_processing import order_tree_model_dict
from utils.benchmarks import branin
from utils.plots import plot_gp_2d
import torch
import matplotlib.pyplot as plt
import numpy as np

class TreeGP(gpy.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.tree_model = None
        self.mean_module = gpy.means.ZeroMean()

        tree_kernel = self._get_tree_kernel(train_inputs, train_targets)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)
    
    def _get_tree_kernel(self, x, y):
        tree_model = lgb.train(
            {"boosting_rounds": 50, "max_depth": 3, "min_data_in_leaf": 1},
            lgb.Dataset(x.numpy(), y.numpy()),
        )
        self.tree_model = tree_model

        original_tree_model_dict = tree_model.dump_model()
        ordered_tree_model_dict = \
            order_tree_model_dict(original_tree_model_dict)

        gbm_model = GbmModel(ordered_tree_model_dict)
        
        return TreeKernel(gbm_model)
    


class TreeKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, gbm_model: GbmModel):
        super().__init__()
        self._gbm_model = gbm_model

        # attributes introduced for leaf optimization
        self._leaf_eval = False
        self._kernel_cache = None
        self._leaf_center = None
        self._var_bnds = None
        self._input_trafo = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        # normal evaluation without any modifications
        np_x1 = x1.detach().numpy()
        np_x2 = x2.detach().numpy()

        out = self._gbm_model.get_gram_mat(np_x1, np_x2)

        if diag:
            out = torch.ones(x1.shape[0])

        return torch.as_tensor(out, dtype=x1.dtype)


def fit_gp(model: TreeGP, likelihood: gpy.likelihoods.Likelihood,
           train_x, train_y):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter:=20):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}"
              f"  \tnoise: {model.likelihood.noise.item():.3f}"
        )
        optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    N_train = 50
    x = torch.rand((N_train, 2)) * 15 + torch.tensor([-5, 0])
    y = branin(x)

    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = TreeGP(x, y, likelihood)

    gp.train()
    likelihood.train()

    fit_gp(gp, likelihood, x, y)

    gp.eval()
    likelihood.eval()

    test_x = torch.meshgrid(torch.linspace(-5, 10, 50), torch.linspace(0, 15, 50), indexing="ij")
    fig, axs = plot_gp_2d(gp, likelihood, x, y, test_x, target=branin)
    fig.savefig("figs/branin_tree_gp.png")

    # plot the covariance matrix
    fig, ax = plt.subplots()
    X, Y = torch.meshgrid(torch.linspace(-5, 10, 10), torch.linspace(0, 15, 10), indexing="ij")
    test_x_stacked = torch.stack([X.flatten(), Y.flatten()], dim=1)
    ax.imshow(gp.covar_module(test_x_stacked).evaluate().detach().numpy())
    plt.show()
