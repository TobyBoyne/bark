from argparse import ArgumentParser

import numpy as np
import torch

from alfalfa.benchmarks import CurrinExp2D
from alfalfa.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.optimizer.information_based_fidelity import (
    propose_fidelity_information_based,
)
from alfalfa.tree_kernels import AlfalfaMOGP, MultitaskGaussianLikelihood

parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="currin")
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
torch.set_default_dtype(torch.float64)

# load black-box function to evaluate
bb_func = CurrinExp2D()

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data([2, 2], args.rnd_seed)
X_train, i_train, y_train = init_data

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
    forest.initialise(bb_func.get_space())
    likelihood = MultitaskGaussianLikelihood(num_tasks=2)

    tree_gp = AlfalfaMOGP(
        (torch.from_numpy(X_train), torch.from_numpy(i_train)),
        torch.from_numpy(y_train),
        likelihood,
        forest,
        num_tasks=2,
    )
    fit_gp_adam(tree_gp)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        bb_func.get_space(), gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(bb_func.get_space(), opt_model, gbm_model, model_core)
    next_x_torch = torch.tensor(next_x).reshape(1, -1)
    next_i = propose_fidelity_information_based(
        tree_gp, next_x_torch, costs=bb_func.costs
    )
    next_y = bb_func(next_x, next_i)

    # update progress
    X_train = np.concatenate((X_train, [next_x]))
    i_train = np.concatenate((i_train, [next_i]))
    y_train = np.concatenate((y_train, next_y))
    print(next_i)
    print(f"{itr}. min_val: {round(min(y_train[i_train == 0]), 5)}")
