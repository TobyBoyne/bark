import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment

from bark.fitting.tree_traversal import get_node_subspace, terminal_nodes


@njit
def _subspace_intersection_volume(sub_u, sub_v):
    lb = np.maximum(sub_u[:, 0], sub_v[:, 0])
    ub = np.minimum(sub_u[:, 1], sub_v[:, 1])
    return np.prod(np.maximum(0, ub - lb))


@njit
def _subspace_intersection_logvolume(sub_u, sub_v):
    lb = np.maximum(sub_u[:, 0], sub_v[:, 0])
    ub = np.minimum(sub_u[:, 1], sub_v[:, 1])
    bound_sizes = np.maximum(0, ub - lb)
    if bound_sizes.min() <= 0:
        return 0
    else:
        return np.sum(np.log(np.maximum(0, ub - lb)))


@njit
def _subspace_product_logvolume(sub_u, sub_v):
    logvolume_u = np.sum(np.log(sub_u[:, 1] - sub_u[:, 0]))
    logvolume_v = np.sum(np.log(sub_v[:, 1] - sub_v[:, 0]))
    return logvolume_u + logvolume_v


@njit
def mutual_information_tree_pair(tree_u, tree_v, bounds, feat_types):
    volume_domain = np.prod(bounds[:, 1] - bounds[:, 0])
    leaves_u, leaves_v = map(terminal_nodes, (tree_u, tree_v))
    mutual_info = 0
    for ui in leaves_u:
        sub_u = get_node_subspace(tree_u, ui, bounds, feat_types)
        for vi in leaves_v:
            sub_v = get_node_subspace(tree_v, vi, bounds, feat_types)
            volume_uv = _subspace_intersection_volume(sub_u, sub_v)

            indep = _subspace_product_logvolume(sub_u, sub_v)
            joint = _subspace_intersection_logvolume(sub_u, sub_v)
            mutual_info += (volume_uv / volume_domain) * (
                joint - indep + np.log(volume_domain)
            )

    return mutual_info


def mutual_information_forest_pair(forest_u, forest_v, bounds, feat_types):
    m = forest_u.shape[0]
    mutual_info_matrix = np.zeros((m, m))
    for i, tree_u in enumerate(forest_u):
        for j, tree_v in enumerate(forest_v):
            mutual_info_matrix[i, j] = mutual_information_tree_pair(
                tree_u, tree_v, bounds, feat_types
            )

    row_ind, col_ind = linear_sum_assignment(mutual_info_matrix, maximize=True)
    return mutual_info_matrix[row_ind, col_ind].sum()
