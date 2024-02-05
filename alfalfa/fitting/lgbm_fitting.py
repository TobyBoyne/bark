"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""
import gpytorch as gpy
import lightgbm as lgb
import torch

from ..forest import AlfalfaForest, AlfalfaTree, DecisionNode, LeafNode


def fit_leaf_gp(model: gpy.models.ExactGP):
    (x,) = model.train_inputs
    y = model.train_targets
    likelihood = model.likelihood

    model.double()
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter := 100):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        if (i + 1) % 100 == 0:
            print(
                "Iter %d/%d - Loss: %.3f  noise: %.3f"
                % (i + 1, training_iter, loss.item(), model.likelihood.noise.item())
            )
        optimizer.step()


def lgbm_to_alfalfa_forest(tree_model: lgb.Booster):
    all_trees = tree_model.dump_model()["tree_info"]

    def get_subtree(node_dict):
        if "leaf_index" in node_dict:
            return LeafNode()
        else:
            var_idx = node_dict["split_feature"]
            threshold = node_dict["threshold"]
            return DecisionNode(
                var_idx=var_idx,
                threshold=threshold,
                left=get_subtree(node_dict["left_child"]),
                right=get_subtree(node_dict["right_child"]),
            )

    trees = [
        AlfalfaTree(root=get_subtree(tree_dict["tree_structure"]))
        for tree_dict in all_trees
    ]
    forest = AlfalfaForest(trees=trees)
    return forest
