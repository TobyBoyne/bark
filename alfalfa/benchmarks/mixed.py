import numpy as np

from ..optimizer.optimizer_utils import get_opt_core
from .base import SynFunc, preprocess_data


class CatAckley(SynFunc):
    """
    adapted from: https://arxiv.org/pdf/1911.12473.pdf"""

    int_idx = {
        0,
    }
    is_vectorised = True

    def __call__(self, x, **kwargs):
        x = np.atleast_2d(x)
        z = x[:, 1:] + x[:, :1]
        return (
            -20 * np.exp(-0.2 * np.sqrt(0.2 * np.sum(z**2)))
            - np.exp(0.2 * np.sum(np.cos(2 * np.pi * z)))
            + 20
            + np.exp(1)
            + x[:, 0]
        )

    @property
    def bounds(self):
        return [[0, 4]] + [[-3.0, 3.0] for _ in range(5)]


class PressureVessel(SynFunc):
    # adapted from: https://www.scielo.br/j/lajss/a/ZsdRkGWRVtDdHJP8WTDFFpB/?format=pdf&lang=en
    is_nonconvex = True

    def __init__(self, seed: int):
        super().__init__(seed)
        self.int_idx = {0, 1}

        def X0(x):
            return x[0] * 0.0625

        def X1(x):
            return x[1] * 0.0625

        self.ineq_constr_funcs = [
            lambda x: -X0(x) + 0.0193 * x[2],
            lambda x: -X1(x) + 0.00954 * x[3],
            lambda x: -np.pi * x[3] * x[2] ** 2 - (4 / 3) * np.pi * x[2] ** 3 + 1296000,
            # this constr. is in the reference but is not necessary
            # lambda x: x[3] - 240
        ]

    def get_model_core(self):
        # define model core
        space = self.space
        model_core = get_opt_core(space)

        # add helper vars
        x = model_core._cont_var_dict

        lb_aux, ub_aux = 1 * 0.0625, 99 * 0.0625
        X0 = model_core.addVar(lb=lb_aux, ub=ub_aux)
        model_core.addConstr(X0 == x[0] * 0.0625)

        X1 = model_core.addVar(lb=lb_aux, ub=ub_aux)
        model_core.addConstr(X1 == x[1] * 0.0625)

        # add constraints
        model_core.addConstr(-X0 + 0.0193 * x[2] <= 0)
        model_core.addConstr(-X1 + 0.00954 * x[3] <= 0)

        # add helper for cubic var
        lb2, ub2 = self.bounds[2]
        x2_squ = model_core.addVar(lb=lb2**2, ub=ub2**2)
        model_core.addConstr(x2_squ == x[2] * x[2])

        model_core.addConstr(
            -np.pi * x[3] * x2_squ - (4 / 3) * np.pi * x[2] * x2_squ + 1296000 <= 0
        )

        # this constr. is in the reference but is not necessary given the bounds
        # model_core.addConstr(x[3] - 240 <= 0)

        # set solver parameter if function is nonconvex
        model_core.Params.LogToConsole = 0
        if self.is_nonconvex:
            model_core.Params.NonConvex = 2

        model_core.update()

        return model_core

    @property
    def bounds(self):
        return [(1, 99), (1, 99), (10.0, 200.0), (10.0, 200.0)]

    @preprocess_data
    def __call__(self, x, **kwargs):
        # true vars X0 and X1 are integer multiples of 0.0625
        def X0(x):
            return x[0] * 0.0625

        def X1(x):
            return x[1] * 0.0625

        f = (
            0.6224 * x[0] * x[2] * x[3]
            + 1.7781 * X1(x) * x[2] ** 2
            + 3.1661 * x[3] * X0(x) ** 2
            + 19.84 * x[2] * X0(x) ** 2
        )
        return f

    @property
    def optimum(self):
        return 6059.715
