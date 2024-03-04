import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import torch

from alfalfa.baselines import RBFGP
from alfalfa.benchmarks import Branin
from alfalfa.utils.plots import plot_gp_nd

bb_func = Branin()

torch.manual_seed(42)
np.random.seed(42)

init_data = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data["X"], init_data["y"]

train_x, train_y = np.asarray(X), np.asarray(y)


def fit_gp(model: RBFGP):
    model.train()
    model.likelihood.train()

    x = model.train_inputs[0]
    y = model.train_targets

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter := 500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        if i % 100 == 0:
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item(),
                )
            )
        optimizer.step()


torch.manual_seed(42)
np.random.seed(42)

init_data = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data["X"], init_data["y"]

train_x, train_y = np.asarray(X), np.asarray(y)

likelihood = gpy.likelihoods.GaussianLikelihood()
gp = RBFGP(torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood)

fit_gp(gp)
gp.eval()
torch.save(gp.state_dict(), "models/branin_rbf_gp.pt")

test_x = torch.meshgrid(
    torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
)
plot_gp_nd(gp, test_x, target=bb_func.vector_apply)
plt.show()
