import numpy as np
import torch
import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt

from alfalfa.leaf_gp.bb_func_utils import get_func
from alfalfa import AlfalfaForest
from alfalfa.tree_models.tree_kernels import AlfalfaGP
from alfalfa.optimizer import get_global_sol, build_opt_model
from alfalfa.leaf_gp.gbm_model import GbmModel
from alfalfa.fitting import BART, BARTTrainParams, BARTData

from alfalfa.utils.plots import plot_gp_1d
from alfalfa.utils.logger import BOLogger

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="hartmann6d")
parser.add_argument("-num-init", type=int, default=2)
parser.add_argument("-num-itr", type=int, default=100)
parser.add_argument("-rnd-seed", type=int, default=101)
parser.add_argument("-solver-type", type=str, default="global") # can also be 'sampling'
parser.add_argument("-has-larger-model", action='store_true')
args = parser.parse_args()

# set random seeds for reproducibility
np.random.seed(args.rnd_seed)
torch.manual_seed((args.rnd_seed))

# load black-box function to evaluate
bb_func = get_func(args.bb_func)

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data(args.num_init, args.rnd_seed)
X, y = init_data['X'], init_data['y']

print(f"* * * initial data targets:")
print("\n".join(f"  val: {yi:.4f}" for yi in y))

# add model_core with constraints if problem has constraints
if bb_func.has_constr():
    model_core = bb_func.get_model_core()
else:
    model_core = None

# main bo loop
print(f"\n* * * start bo loop...")
for itr in range(args.num_itr):
    X_train, y_train = np.asarray(X), np.asarray(y)

    
    forest = AlfalfaForest(height=1, num_trees=5)
    forest.initialise(bb_func.get_space())

    likelihood = gpy.likelihoods.GaussianLikelihood()
    tree_gp = AlfalfaGP(torch.from_numpy(X_train), torch.from_numpy(y_train), likelihood, forest)

    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, tree_gp)
    train_params = BARTTrainParams(
        warmup_steps=50,
        n_steps=50
    )
    data = BARTData(bb_func.get_space(), X_train)
    bart = BART(tree_gp, data, train_params)
    bart.run()
    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(bb_func.get_space(), gbm_model, tree_gp, 1.96)
    var_bnds, next_x, curr_mean, curr_var = get_global_sol(
        bb_func.get_space(), opt_model, gbm_model
    )
    next_y = bb_func(next_x)

    # update progress
    X.append(next_x)
    y.append(next_y)

    print(f"{itr}. min_val: {round(min(y), 5)}")