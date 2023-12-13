import torch
import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt

from alfalfa.utils.benchmarks import rescaled_branin, branin
from alfalfa.tree_models.lgbm_tree import lgbm_to_alfalfa_forest
from alfalfa.tree_models.tree_kernels import AFGP
from alfalfa.utils.plots import plot_gp_2d

def fit_gp(x, y, model, likelihood):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter:=200):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        if (i+1) % 100 == 0:
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    N_train = 50
    x = torch.rand((N_train, 2))
    # x = torch.rand((N_train, 2)) * 15 + torch.tensor([-5, 0])

    f = rescaled_branin(x)
    # f = branin(x)
    y = f

    noise_var = 0.2
    y = f + torch.randn_like(f) * noise_var ** 0.5

    tree_model = lgb.train(
                {"max_depth": 2, "min_data_in_leaf": 1},
                lgb.Dataset(x.numpy(), y.numpy()),
                num_boost_round=100
            )

    forest = lgbm_to_alfalfa_forest(tree_model)
    likelihood = gpy.likelihoods.GaussianLikelihood()

    gp = AFGP(x, y, likelihood, forest)
    fit_gp(x, y, gp, likelihood)
    gp.eval()
    # torch.save(gp.state_dict(), "models/branin_leaf_gp.pt")

    test_x = torch.meshgrid(torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij")
    plot_gp_2d(gp, likelihood, x, y, test_x, target=rescaled_branin)
    plt.show()

    # test_x = torch.meshgrid(torch.linspace(-5, 10, 100), torch.linspace(0, 15, 100), indexing="ij")
    # fig, axs = plot_gp_2d(gp, likelihood, x, y, test_x, target=branin)
    # plt.show()
