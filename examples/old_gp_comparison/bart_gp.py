import gpytorch as gpy
import matplotlib.pyplot as plt
import scipy.stats as stats
from jaxtyping import install_import_hook

with install_import_hook("alfalfa", "beartype.beartype"):
    from bark.fitting import BART, BARTData, BARTTrainParams
    from bark.forest import AlfalfaForest
    from bark.tree_kernels import AlfalfaGP, AlfalfaMixtureModel
    from bark.utils.metrics import nlpd

import problem

if __name__ == "__main__":
    likelihood = gpy.likelihoods.GaussianLikelihood(
        noise_constraint=gpy.constraints.Positive()
    )
    forest = AlfalfaForest(height=0, num_trees=20)
    space = problem.bb_func.space
    forest.initialise(space)
    gp = AlfalfaGP(problem.train_x_torch, problem.train_y_torch, likelihood, forest)

    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    output = gp(problem.train_x_torch)
    loss = -mll(output, problem.train_y_torch)
    print(f"Initial loss={loss}")

    data = BARTData(space, problem.train_x_np)
    params = BARTTrainParams(
        # alpha=0.95,
        warmup_steps=5,
        n_steps=5,
        lag=1,
    )
    bart = BART(
        gp,
        data,
        params,
        scale_prior=stats.halfnorm(scale=1.0),
        noise_prior=stats.halfnorm(scale=1.0),
    )
    logger = bart.run_multichain(n_chains=4)
    # logger = bart.run()

    gp.eval()

    output = gp.likelihood(gp(problem.test_x_torch))
    test_loss = nlpd(output, problem.test_y_torch, diag=False)
    print(f"GP test loss={test_loss}")

    sampled_model = AlfalfaMixtureModel(
        problem.train_x_torch,
        problem.train_y_torch,
        logger["samples"],
        space,
        sampling_seed=100,
    )

    output = sampled_model(problem.test_x_torch, predict_y=True)
    test_loss = nlpd(output, problem.test_y_torch, diag=False)
    print(f"Full mixture test loss={test_loss}")

    for gp in sampled_model.gp_samples_iter():
        output = gp.likelihood(gp(problem.test_x_torch))
        test_loss = nlpd(output, problem.test_y_torch, diag=False)
        print(f"Sampled test loss={test_loss}")

    # continue chain
    num_samples = len(logger["samples"])
    step = num_samples // 4
    states = logger["samples"][step - 1 :: step]
    logger = bart.run_multichain(n_chains=4, initial_states=states)

    # test_x = torch.meshgrid(
    #     torch.linspace(0, 1, 25), torch.linspace(0, 1, 25), indexing="ij"
    # )

    # plot_gp_nd(gp, test_x, target=problem.bb_func.vector_apply)
    # plot_gp_nd(sampled_model, test_x, target=problem.bb_func.vector_apply)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(logger["noise"])
    axs[1].plot(logger["scale"])
    plt.show()
