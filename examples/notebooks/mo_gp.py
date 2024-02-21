import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt
import torch

from alfalfa.fitting import fit_leaf_gp, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaMOGP
from alfalfa.utils.space import Space

torch.set_default_dtype(torch.float64)


train_x1 = torch.rand(50)
train_x2 = torch.rand(10)

train_y1 = torch.sin(train_x1 * (2 * torch.pi)) + torch.randn(train_x1.size()) * 0.2
train_y2 = torch.cos(train_x2 * (2 * torch.pi)) + torch.randn(train_x2.size()) * 0.2

# likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
likelihood = gpy.likelihoods.GaussianLikelihood()

tree_model = lgb.train(
    {"max_depth": 3, "min_data_in_leaf": 1},
    lgb.Dataset(train_x1.numpy()[:, None], train_y1.numpy()),
    num_boost_round=50,
)

forest = lgbm_to_alfalfa_forest(tree_model)
forest.initialise(Space([[0.0, 1.0]]))

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2])

# Here we have two iterms that we're passing in as train_inputs
model = AlfalfaMOGP((full_train_x, full_train_i), full_train_y, likelihood, forest)

fit_leaf_gp(model)

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Test points every 0.02 in [0,1]
test_x = torch.linspace(0, 1, 51)
test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpy.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    observed_pred_y2 = likelihood(model(test_x, test_i_task2))


# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), "k*")
    # Predictive mean as blue line
    ax.plot(test_x.detach().numpy(), rand_var.mean.detach().numpy(), "b")
    # Shade in confidence
    ax.fill_between(
        test_x.detach().numpy(),
        lower.detach().numpy(),
        upper.detach().numpy(),
        alpha=0.5,
    )
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_title(title)


# Plot both tasks
ax_plot(y1_ax, train_y1, train_x1, observed_pred_y1, "Observed Values (Likelihood)")
ax_plot(y2_ax, train_y2, train_x2, observed_pred_y2, "Observed Values (Likelihood)")
plt.show()
