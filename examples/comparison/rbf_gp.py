import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.gps import RBFGP
from alfalfa.utils.benchmarks import rescaled_branin
from alfalfa.utils.plots import plot_gp_2d
    
def fit_gp(x, y, model, likelihood):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter:=500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        if i % 100 == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

if __name__ == "__main__":
    torch.manual_seed(42)
    N_train = 50
    x = torch.rand((N_train, 2))
    f = rescaled_branin(x)

    y = f + torch.randn_like(f) * 0.2**0.5

    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = RBFGP(x, y, likelihood)

    fit_gp(x, y, gp, likelihood)
    gp.eval()
    torch.save(gp.state_dict(), "models/branin_rbf_gp.pt")

    test_x = torch.meshgrid(torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij")
    plot_gp_2d(gp, likelihood, x, y, test_x, target=rescaled_branin)
    plt.show()