from argparse import ArgumentParser

import gpytorch as gpy
import numpy as np
import torch

from alfalfa.benchmarks import map_benchmark
from alfalfa.fitting import BART, BARTData, BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.tree_kernels import AlfalfaGP

torch.set_default_dtype(torch.float64)


parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="hartmann6d")
parser.add_argument("-num-init", type=int, default=5)
parser.add_argument("-num-itr", type=int, default=50)
parser.add_argument("-rnd-seed", type=int, default=101)
parser.add_argument(
    "-solver-type", type=str, default="global"
)  # can also be 'sampling'
parser.add_argument("-has-larger-model", action="store_true")
parser.add_argument("-outfile", type=str, default="")
args = parser.parse_args()

# set random seeds for reproducibility
np.random.seed(args.rnd_seed)
torch.manual_seed((args.rnd_seed))

# load black-box function to evaluate
bb_func = map_benchmark(args.bb_func, seed=args.rnd_seed)
space = bb_func.space

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
X_train, y_train = bb_func.get_init_data(args.num_init, args.rnd_seed)

print("* * * initial data targets:")
print("\n".join(f"  val: {yi:.4f}" for yi in y_train))

# add model_core with constraints if problem has constraints
model_core = bb_func.get_model_core()

# define trees outside of BO loop
# allows for warm-up steps to be re-used across iterations
forest = AlfalfaForest(height=0, num_trees=10)
forest.initialise(space)


# main bo loop
print("\n* * * start bo loop...")
for itr in range(args.num_itr):
    likelihood = gpy.likelihoods.GaussianLikelihood()
    tree_gp = AlfalfaGP(
        torch.from_numpy(X_train), torch.from_numpy(y_train), likelihood, forest
    )
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, tree_gp)
    train_params = BARTTrainParams(warmup_steps=10 if itr == 0 else 10, n_steps=1)
    data = BARTData(space, X_train)
    bart = BART(tree_gp, data, train_params, noise_prior=None, scale_prior=None)
    bart.run()
    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        space, gbm_model, tree_gp, kappa=1.96, model_core=model_core
    )
    next_x = propose(space, opt_model, gbm_model, model_core)
    next_y = bb_func(next_x)

    # update progress
    X_train = np.concatenate((X_train, [next_x]))
    y_train = np.concatenate((y_train, [next_y]))

    print(f"{itr}. min_val: {round(min(y_train), 5)}")

if args.outfile:
    with open(args.outfile, "w+") as f:
        np.save(f, y_train)
