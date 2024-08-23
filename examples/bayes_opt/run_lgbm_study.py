from argparse import ArgumentParser

import gpytorch as gpy
import numpy as np
import torch

from bark.benchmarks import map_benchmark
from bark.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from bark.optimizer import build_opt_model, propose
from bark.optimizer.gbm_model import GbmModel
from bark.tree_kernels import AlfalfaGP

parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="branin")
parser.add_argument("-num-init", type=int, default=5)
parser.add_argument("-num-itr", type=int, default=100)
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
bb_func = map_benchmark(args.bb_func, seed=42)
# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data(args.num_init, args.rnd_seed)
X_train, y_train = init_data

print("* * * initial data targets:")
print("\n".join(f"  val: {yi:.4f}" for yi in y_train))

# add model_core with constraints if problem has constraints
model_core = bb_func.get_model_core()

# modify tree model hyperparameters
if not args.has_larger_model:
    tree_params = {"boosting_rounds": 50, "max_depth": 3, "min_data_in_leaf": 1}
else:
    tree_params = {"boosting_rounds": 100, "max_depth": 5, "min_data_in_leaf": 1}

# main bo loop
print("\n* * * start bo loop...")
for itr in range(args.num_itr):
    booster = fit_lgbm_forest(X_train, y_train)
    forest = lgbm_to_alfalfa_forest(booster)
    space = bb_func.space
    forest.initialise(space)
    likelihood = gpy.likelihoods.GaussianLikelihood()
    tree_gp = AlfalfaGP(
        torch.from_numpy(X_train), torch.from_numpy(y_train), likelihood, forest
    )
    fit_gp_adam(tree_gp)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(space, gbm_model, tree_gp, 1.96, model_core=model_core)
    next_x = propose(space, opt_model, gbm_model, model_core)
    next_y = bb_func(next_x)

    # update progress
    X_train = np.concatenate((X_train, [next_x]))
    y_train = np.concatenate((y_train, [next_y]))

    print(f"{itr}. min_val: {round(min(y_train), 5)}")
