import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt

from bark import enums, types

smooth_indicator = jax.lax.logistic


@jax.jit
def _pass_one_through_soft_tree(
    X: Float[Array, " d"],
    tree: types.SoftTree,
    feat_types: types.FeatTypesT,
) -> Float[Array, " max_nodes"]:
    # at every node, compute the splits
    cont_splits = smooth_indicator((X[tree.feature_idx] - tree.threshold) / tree.tau)

    # propogate the splits down the tree
    def loop(cum_indicator: Float[Array, " max_nodes"], d: UInt[Array, ""]):
        nodes_at_depth = cum_indicator[2**d - 1 : 2 ** (d + 1) - 1]
        cum_indicator = cum_indicator.at[2 ** (d + 1) - 1 : 2 ** (d + 2) - 1 : 2].mul(
            nodes_at_depth
        )
        cum_indicator = cum_indicator.at[2 ** (d + 1) : 2 ** (d + 2) - 1 : 2].mul(
            1 - nodes_at_depth
        )
        return cum_indicator, None

    cum_indicator = cont_splits
    depths = jnp.arange(enums.MAX_DEPTH)
    cum_indicator, _ = jax.lax.scan(
        loop, cum_indicator, depths, enums.MAX_DEPTH - 1, unroll=enums.MAX_DEPTH - 1
    )
    cum_indicator.at[tree.feature_idx == enums.NodeState.Inactive].set(0.0)

    return cum_indicator
