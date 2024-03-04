import numpy as np

from ..optimizer.optimizer_utils import get_opt_core
from .base import SynFunc


class G1(SynFunc):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page506.htm
    def __init__(self):
        super().__init__()
        self.ineq_constr_funcs = [
            lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,  # g1
            lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,  # g2
            lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,  # g3
            lambda x: -8 * x[0] + x[9],  # g4
            lambda x: -8 * x[1] + x[10],  # g5
            lambda x: -3 * x[2] + x[11],  # g6
            lambda x: -2 * x[3] - x[4] + x[9],  # g7
            lambda x: -2 * x[5] - x[6] + x[10],  # g8
            lambda x: -2 * x[7] - x[8] + x[11],  # g9
        ]

    def get_bounds(self):
        bnds = []
        for idx in range(13):
            lb = 0.0
            ub = 1.0 if idx not in (9, 10, 11) else 100.0
            bnds.append((lb, ub))
        return bnds

    def __call__(self, x, **kwargs):
        f = (
            5 * sum(x[i] for i in range(4))
            - 5 * sum(x[i] ** 2 for i in range(4))
            - sum(x[i] for i in range(4, 13))
        )
        return f


class G3(SynFunc):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2613.htm
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim
        self.is_nonconvex = True
        self.eq_constr_funcs = [
            lambda x: sum([x[i] * x[i] for i in range(self.dim)]) - 1  # h1
        ]

    def __call__(self, x, **kwargs):
        f = (np.sqrt(self.dim) ** self.dim) * np.prod([x[i] for i in range(self.dim)])
        f = -float(f)
        return f

    def get_bounds(self):
        return [(0.0, 1.0) for _ in range(self.dim)]


class G4(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        def u(x):
            return (
                85.334407
                + 0.0056858 * x[1] * x[4]
                + 0.0006262 * x[0] * x[3]
                - 0.0022053 * x[2] * x[4]
            )

        def v(x):
            return (
                80.51249
                + 0.0071317 * x[1] * x[4]
                + 0.0029955 * x[0] * x[1]
                + 0.0021813 * x[2] ** 2
            )

        def w(x):
            return (
                9.300961
                + 0.0047026 * x[2] * x[4]
                + 0.0012547 * x[0] * x[2]
                + 0.0019085 * x[2] * x[3]
            )

        self.ineq_constr_funcs = [
            lambda x: -u(x),  # g1
            lambda x: u(x) - 92.0,  # g2
            lambda x: -v(x) + 90.0,  # g3
            lambda x: v(x) - 110.0,  # g4
            lambda x: -w(x) + 20.0,  # g5
            lambda x: w(x) - 25.0,  # g6
        ]

    def get_bounds(self):
        lb = [78.0, 33.0, 27.0, 27.0, 27.0]
        ub = [102.0, 45.0, 45.0, 45.0, 45.0]
        return [(lb[idx], ub[idx]) for idx in range(5)]

    def __call__(self, x, **kwargs):
        f = (
            5.3578547 * x[2] ** 2
            + 0.8356891 * x[0] * x[4]
            + 37.293239 * x[0]
            - 40792.141
        )
        return f


class G6(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        self.ineq_constr_funcs = [
            lambda x: -((x[0] - 5) ** 2) - (x[1] - 5) ** 2 + 100.0,
            lambda x: (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81,
        ]

    def get_bounds(self):
        return [(13.0, 100.0), (0.0, 100.0)]

    def __call__(self, x, **kwargs):
        f = (x[0] - 10.0) ** 3 + (x[1] - 20.0) ** 3
        return f


class G7(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.ineq_constr_funcs = [
            lambda x: 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7] - 105,
            lambda x: 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7],
            lambda x: -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12,
            lambda x: 3 * (x[0] - 2) ** 2
            + 4 * (x[1] - 3) ** 2
            + 2 * x[2] ** 2
            - 7 * x[3]
            - 120,
            lambda x: 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40,
            lambda x: 0.5 * (x[0] - 8) ** 2
            + 2 * (x[1] - 4) ** 2
            + 3 * x[4] ** 2
            - x[5]
            - 30,
            lambda x: x[0] ** 2
            + 2 * (x[1] - 2) ** 2
            - 2 * x[0] * x[1]
            + 14 * x[4]
            - 6 * x[5],
            lambda x: -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9],
        ]

    def get_bounds(self):
        return [(-10.0, 10.0) for _ in range(10)]

    def __call__(self, x, **kwargs):
        f = (
            x[0] ** 2
            + x[1] ** 2
            + x[0] * x[1]
            - 14 * x[0]
            - 16 * x[1]
            + (x[2] - 10) ** 2
            + 4 * (x[3] - 5) ** 2
            + (x[4] - 3) ** 2
            + 2 * (x[5] - 1) ** 2
            + 5 * x[6] ** 2
            + 7 * (x[7] - 11) ** 2
            + 2 * (x[8] - 10) ** 2
            + (x[9] - 7) ** 2
            + 45
        )
        return f


class G10(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        self.ineq_constr_funcs = [
            lambda x: -1 + 0.0025 * (x[3] + x[5]),
            lambda x: -1 + 0.0025 * (-x[3] + x[4] + x[6]),
            lambda x: -1 + 0.01 * (-x[4] + x[7]),
            lambda x: 100 * x[0] - x[0] * x[5] + 833.33252 * x[3] - 83333.333,
            lambda x: x[1] * x[3] - x[1] * x[6] - 1250 * x[3] + 1250 * x[4],
            lambda x: x[2] * x[4] - x[2] * x[7] - 2500 * x[4] + 1250000,
        ]

    def get_bounds(self):
        lb = [100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        ub = [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        return [(lb[idx], ub[idx]) for idx in range(8)]

    def __call__(self, x, **kwargs):
        f = x[0] + x[1] + x[2]
        return f


class Alkylation(SynFunc):
    # original source: R. N. Sauer, A. R. Colville and C. W. Bunvick,
    #                  ‘Computer points the way to more profits’, Hydrocarbon Process.
    #                  Petrol. Refiner. 43,8492 (1964).
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py

    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        def X1(x):
            return x[0]

        def X2(x):
            return x[1]

        def X3(x):
            return x[2]

        def X4(x):
            return x[3]

        def X5(x):
            return x[4]

        def X6(x):
            return x[5]

        def X7(x):
            return x[6]

        def x5(x):
            return 1.22 * X4(x) - X1(x)

        def x6(x):
            return (98000.0 * X3(x)) / (X4(x) * X6(x) + 1000.0 * X3(x))

        def x8(x):
            return (X2(x) + x5(x)) / X1(x)

        self.ineq_constr_funcs = [
            lambda x: 0.99 * X4(x)
            - (X1(x) * (1.12 + 0.13167 * x8(x) - 0.00667 * x8(x) ** 2)),
            lambda x: (X1(x) * (1.12 + 0.13167 * x8(x) - 0.00667 * x8(x) ** 2))
            - (100.0 / 99.0) * X4(x),
            lambda x: 0.99 * X5(x)
            - (86.35 + 1.098 * x8(x) - 0.038 * x8(x) ** 2 + 0.325 * (x6(x) - 89.0)),
            lambda x: (
                86.35 + 1.098 * x8(x) - 0.038 * x8(x) ** 2 + 0.325 * (x6(x) - 89.0)
            )
            - (100.0 / 99.0) * X5(x),
            lambda x: 0.9 * X6(x) - (35.82 - 0.222 * X7(x)),
            lambda x: (35.82 - 0.222 * X7(x)) - (10.0 / 9.0) * X6(x),
            lambda x: 0.99 * X7(x) - (-133 + 3 * X5(x)),
            lambda x: (-133 + 3.0 * X5(x)) - (100.0 / 99.0) * X7(x),
            lambda x: x5(x) - 2000,
            lambda x: -x5(x),
            lambda x: x6(x) - 93.0,
            lambda x: 85.0 - x6(x),
            lambda x: x8(x) - 12.0,
            lambda x: 3.0 - x8(x),
        ]

    def is_feas(self, x):
        # alkylation can have division by zero error
        if not self.has_constr():
            return True

        # check if any constraint is above feasibility threshold
        for val in self.get_feas_vals(x):
            if val is None or val > 1e-5:
                return False
        return True

    def get_feas_ineq_vals(self, x):
        # compute individual feasibility vals for all constr.
        ## check division by zero error for Alkylation bb_func since that can occur

        try:
            return super().get_feas_ineq_vals(x)

        except ZeroDivisionError:
            return [None]

    def get_feas_penalty(self, x):
        # compute squared penalty of constr. violation vals
        if not self.has_constr():
            return 0.0

        feas_penalty = 0.0
        for vals in self.get_feas_vals(x):
            # vals can be None if 'ZeroDivisionError' is encountered
            #   return maximum penalty + 500 for this case
            if vals is None:
                return max(self.y_penalty) + 500

            feas_penalty += vals**2
        return feas_penalty

    def get_model_core(self):
        # define model core
        space = self.get_space()
        model_core = get_opt_core(space)

        # add helper vars
        x = model_core._cont_var_dict

        lb, ub = self.get_lb(), self.get_ub()

        # add x5 constr
        x5 = model_core.addVar(lb=0.0, ub=2000.0)
        model_core.addConstr(x5 == 1.22 * x[3] - x[0])

        # add x6 constrs
        x35 = model_core.addVar(lb=lb[3] * lb[5], ub=ub[3] * ub[5])
        model_core.addConstr(x35 == x[3] * x[5])

        x6 = model_core.addVar(lb=85.0, ub=93.0)
        model_core.addConstr(x6 * x35 + 1000.0 * x[2] * x6 == 98000.0 * x[2])

        # add x8 constrs
        x8 = model_core.addVar(lb=3.0, ub=12.0)
        model_core.addConstr(x8 * x[0] == x[1] + x5)
        model_core.addConstr(x[0] >= 0.1)

        squ_x8 = model_core.addVar(lb=3.0**2, ub=12.0**2)
        model_core.addConstr(squ_x8 == x8 * x8)

        # add other constrs
        model_core.addConstr(
            0.99 * x[3] - (x[0] * (1.12 + 0.13167 * x8 - 0.00667 * squ_x8)) <= 0.0
        )
        model_core.addConstr(
            (x[0] * (1.12 + 0.13167 * x8 - 0.00667 * squ_x8)) - (100.0 / 99.0) * x[3]
            <= 0.0
        )
        model_core.addConstr(
            0.99 * x[4] - (86.35 + 1.098 * x8 - 0.038 * squ_x8 + 0.325 * (x6 - 89.0))
            <= 0.0
        )
        model_core.addConstr(
            (86.35 + 1.098 * x8 - 0.038 * squ_x8 + 0.325 * (x6 - 89.0))
            - (100.0 / 99.0) * x[4]
            <= 0.0
        )
        model_core.addConstr(0.9 * x[5] - (35.82 - 0.222 * x[6]) <= 0.0)
        model_core.addConstr((35.82 - 0.222 * x[6]) - (10.0 / 9.0) * x[5] <= 0.0)
        model_core.addConstr(0.99 * x[6] - (-133 + 3 * x[4]) <= 0.0)
        model_core.addConstr((-133 + 3.0 * x[4]) - (100.0 / 99.0) * x[6] <= 0.0)

        # set solver parameter if function is nonconvex
        model_core.Params.LogToConsole = 0
        if self.is_nonconvex:
            model_core.Params.NonConvex = 2

        model_core.update()
        return model_core

    def get_bounds(self):
        lb = [0.0, 0.0, 0.0, 0.0, 90.0, 0.01, 145.0]
        ub = [2000.0, 16000.0, 120.0, 5000.0, 95.0, 4.0, 162.0]
        return [(lb[idx], ub[idx]) for idx in range(7)]

    def __call__(self, x, **kwargs):
        X1 = x[0]
        X2 = x[1]
        X3 = x[2]
        X4 = x[3]
        X5 = x[4]
        x5 = 1.22 * X4 - X1
        f = -(0.063 * X4 * X5 - 5.04 * X1 - 0.035 * X2 - 10.0 * X3 - 3.36 * x5)
        return f
