import gpytorch as gpy
import matplotlib.pyplot as plt
import problem
import torch

from alfalfa.baselines import RBFGP
from alfalfa.fitting import fit_gp_adam
from alfalfa.utils.plots import plot_gp_nd

likelihood = gpy.likelihoods.GaussianLikelihood()
gp = RBFGP(problem.train_x_torch, problem.train_y_torch, likelihood)

fit_gp_adam(gp)
gp.eval()
torch.save(gp.state_dict(), "models/branin_rbf_gp.pt")

test_x = torch.meshgrid(
    torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
)
plot_gp_nd(gp, test_x, target=problem.bb_func.vector_apply)
plt.show()
