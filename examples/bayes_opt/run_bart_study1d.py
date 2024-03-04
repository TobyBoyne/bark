from argparse import ArgumentParser

import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from alfalfa.benchmarks import map_benchmark
from alfalfa.fitting import BART, BARTData, BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.logger import BOLogger

torch.set_default_dtype(torch.float64)


parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="himmelblau1d")
parser.add_argument("-num-init", type=int, default=2)
parser.add_argument("-num-itr", type=int, default=10)
parser.add_argument("-rnd-seed", type=int, default=101)
parser.add_argument(
    "-solver-type", type=str, default="global"
)  # can also be 'sampling'
parser.add_argument("-has-larger-model", action="store_true")
args = parser.parse_args()

# set random seeds for reproducibility
np.random.seed(args.rnd_seed)
torch.manual_seed((args.rnd_seed))

# load black-box function to evaluate
bb_func = map_benchmark(args.bb_func)

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data(args.num_init, args.rnd_seed)
X, y = init_data["X"], init_data["y"]

print("* * * initial data targets:")
print("\n".join(f"  val: {yi:.4f}" for yi in y))

# add model_core with constraints if problem has constraints
if bb_func.has_constr():
    model_core = bb_func.get_model_core()
else:
    model_core = None

test_x = torch.linspace(0, 1, 50)
logger = BOLogger(X, y, target=lambda x: bb_func(x.reshape(1, -1)))

train_params = BARTTrainParams(warmup_steps=100, n_steps=50)


# main bo loop
print("\n* * * start bo loop...")
for itr in range(args.num_itr):
    X_train, y_train = np.asarray(X), np.asarray(y)

    forest = AlfalfaForest(height=0, num_trees=10)
    forest.initialise(bb_func.get_space())

    likelihood = gpy.likelihoods.GaussianLikelihood()
    tree_gp = AlfalfaGP(
        torch.from_numpy(X_train), torch.from_numpy(y_train), likelihood, forest
    )

    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, tree_gp)
    data = BARTData(bb_func.get_space(), X_train)
    bart = BART(tree_gp, data, train_params)
    bart.run()
    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(bb_func.get_space(), gbm_model, tree_gp, 1.96)
    next_x = propose(bb_func.get_space(), opt_model, gbm_model)
    next_y = bb_func(next_x)

    # update progress
    X.append(next_x)
    y.append(next_y)

    print(f"{itr}. min_val: {round(min(y), 5)}")
    print(f"{next_x[0]:.4f}, {next_y:.4f}")

    tree_gp.eval()
    with torch.no_grad():
        observed_pred = likelihood(tree_gp(test_x))
        mean = observed_pred.mean.numpy()
        confidence = [c.numpy() for c in observed_pred.confidence_region()]

    logger.log_bo_step(
        gp_mean=mean, gp_confidence=confidence, new_train_x=next_x, new_train_y=next_y
    )
    tree_gp.train()

fig, ax = plt.subplots()
ax.set_ylim(-3, 3)
ani = FuncAnimation(
    fig,
    lambda f: logger.plot_bo_step(ax, f, test_x.numpy()),
    frames=logger.num_bo_steps,
    interval=1000,
)
ani.save("figs/bo_bart_1d_animated.mp4")
plt.show()
