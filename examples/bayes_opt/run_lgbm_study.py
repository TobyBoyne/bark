from argparse import ArgumentParser

import gpytorch as gpy
import lightgbm as lgb
import numpy as np
import torch
from botorch.utils import standardize

from alfalfa.fitting.lgbm_fitting import fit_leaf_gp, lgbm_to_alfalfa_forest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.bb_funcs import get_func

parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="g3")
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
bb_func = get_func(args.bb_func)

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data(args.num_init, args.rnd_seed)
X, y = init_data["X"], init_data["y"]

print("* * * initial data targets:")
print("\n".join(f"  val: {yi:.4f}" for yi in y))

# add model_core with constraints if problem has constraints
model_core = bb_func.get_model_core()

# modify tree model hyperparameters
if not args.has_larger_model:
    tree_params = {"boosting_rounds": 50, "max_depth": 3, "min_data_in_leaf": 1}
else:
    tree_params = {"boosting_rounds": 100, "max_depth": 5, "min_data_in_leaf": 1}

X = [
    [
        0.3852022485955958,
        0.6228813018628303,
        0.41400182506057703,
        0.1737091811912213,
        0.5119263815690878,
    ],
    [
        0.6309613412236432,
        0.3389909693535073,
        0.3890018127044594,
        0.09168999687470201,
        0.5720542609923439,
    ],
    [
        0.026754203940608876,
        0.8159070353313401,
        0.16546117940401706,
        0.5515269095819895,
        0.04498464003309791,
    ],
    [
        0.13132637524066224,
        0.550525185688497,
        0.5990280388082755,
        0.5565417806821193,
        0.10537718328798747,
    ],
    [
        0.550303207113237,
        0.15288904447725513,
        0.7759863478354929,
        0.22143391155370426,
        0.15035409357404353,
    ],
    [
        0.20981060772125404,
        0.447785311849383,
        0.4475563508096775,
        0.4478649921592921,
        0.5954669104928565,
    ],
    [
        0.43821763086734405,
        0.5069081165796212,
        0.4842215099799958,
        0.33819639508458027,
        0.4496263723984085,
    ],
    [
        0.41366927780220014,
        0.4143023738351165,
        0.5931916495543298,
        0.4140607570139238,
        0.3659393188807731,
    ],
    [
        0.2974964653876007,
        0.586693269783146,
        0.507825824258923,
        0.09931892819369466,
        0.5472984040865184,
    ],
    [
        0.4910719738160204,
        0.5685992277358216,
        0.3447276996850411,
        0.49320739595445195,
        0.2710238371280612,
    ],
]
y = [
    -0.4938006512659011,
    -0.24396453542971938,
    -0.005009389420974978,
    -0.1419860476620413,
    -0.12151136242104588,
    -0.6268660796903548,
    -0.9143425502359379,
    -0.8611203975950393,
    -0.2693325948728151,
    -0.719266240868719,
]


# main bo loop
print("\n* * * start bo loop...")
for itr in range(args.num_itr):
    X_train, y_train = np.asarray(X), np.asarray(y)
    y_train_torch = standardize(torch.from_numpy(y_train)).double()

    tree_model = lgb.train(
        {"max_depth": 3, "min_data_in_leaf": 1, "verbose": -1},
        lgb.Dataset(X_train, y_train, params={"verbose": -1}),
        num_boost_round=50,
    )
    forest = lgbm_to_alfalfa_forest(tree_model)
    forest.initialise(bb_func.get_space())
    likelihood = gpy.likelihoods.GaussianLikelihood(
        noise_constraint=gpy.constraints.Interval(5e-4, 0.2)
    )
    tree_gp = AlfalfaGP(torch.from_numpy(X_train), y_train_torch, likelihood, forest)
    fit_leaf_gp(tree_gp)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        bb_func.get_space(), gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(bb_func.get_space(), opt_model, gbm_model, model_core)
    next_y = bb_func(next_x)

    # tree_gp.eval()
    # test_x = torch.linspace(0, 1, 100).reshape(-1, 1)
    # target = lambda x: torch.tensor([bb_func(x[i, :]) for i in range(test_x.shape[0])])
    # plot_gp_1d(tree_gp, test_x, target)
    # plt.show()

    # update progress
    X.append(next_x)
    y.append(next_y)

    print(f"{itr}. min_val: {round(min(y), 5)}")
