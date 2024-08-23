import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt
import torch

from bark.fitting import fit_gp_adam, lgbm_to_alfalfa_forest
from bark.tree_kernels import AlfalfaMOGP, MultitaskGaussianLikelihood
from bark.utils.space import Space

plt.style.use(["science", "no-latex", "grid"])


torch.set_default_dtype(torch.float64)
torch.random.manual_seed(42)


train_x1 = torch.rand(20)
train_x2 = torch.rand(10) * 0.5

train_y1 = torch.sin(train_x1 * (2 * torch.pi)) + torch.randn(train_x1.size()) * 0.5
train_y2 = torch.sin(train_x2 * (2 * torch.pi)) + torch.randn(train_x2.size()) * 0.01

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2])


# likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=2, rank=0)
# likelihood = gpy.likelihoods.GaussianLikelihood()
likelihood = MultitaskGaussianLikelihood(num_tasks=2)

tree_model = lgb.train(
    {"max_depth": 3, "min_data_in_leaf": 1},
    lgb.Dataset(train_x1.numpy()[:, None], train_y1.numpy()),
    num_boost_round=50,
)

forest = lgbm_to_alfalfa_forest(tree_model)
forest.initialise(Space([[0.0, 1.0]]))


# Here we have two iterms that we're passing in as train_inputs
model = AlfalfaMOGP((full_train_x, full_train_i), full_train_y, likelihood, forest)

fit_gp_adam(model)

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(6, 3))

# Test points every 0.02 in [0,1]
test_x = torch.linspace(0, 1, 78)
test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpy.settings.fast_pred_var():
    observed_pred_y1 = likelihood(
        model(test_x, test_i_task1),
        [
            test_i_task1,
        ],
    )
    observed_pred_y2 = likelihood(
        model(test_x, test_i_task2),
        [
            test_i_task2,
        ],
    )


# Define plotting function
def ax_plot(ax: plt.Axes, train_y, train_x, rand_var, title):
    ax.scatter(
        train_x.detach().numpy(), train_y.detach().numpy(), marker="x", c="black"
    )

    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    # Predictive mean as blue line
    ax.plot(
        test_x.detach().numpy(),
        rand_var.mean.detach().numpy(),
        color="C0",
        label="Mean",
    )
    # Shade in confidence
    ax.fill_between(
        test_x.detach().numpy(),
        lower.detach().numpy(),
        upper.detach().numpy(),
        alpha=0.5,
        label="Confidence",
        color="C0",
    )
    ax.set_ylim([-3, 3])

    ax.plot(
        test_x.detach().numpy(),
        torch.sin(test_x * (2 * torch.pi)).detach().numpy(),
        color="C2",
        label="True Function",
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlim(0, 1)
    # ax.legend()


# Plot both tasks
ax_plot(y1_ax, train_y1, train_x1, observed_pred_y1, "Low fidelity")
ax_plot(y2_ax, train_y2, train_x2, observed_pred_y2, "High fidelity")
fig.savefig("figs/mo_gp_nomodel.png")
plt.show()
