import numpy as np
from .optimizer_utils import conv2list

def get_func(bb_name):
    # local vs. global acquisition optimization
    # syn_func = {
    #     "hartmann6d": Hartmann6D,
    # }.get(bb_name)

    # if syn_func is None:
    #     raise KeyError(f"'{bb_name}' is not a valid 'bb_name'!")
    
    # return syn_func()
    if bb_name == 'hartmann6d':
        return Hartmann6D()
    elif bb_name == "himmelblau1d":
        return Himmelblau1D()
    elif bb_name == 'rastrigin':
        return Rastrigin()
    elif bb_name == 'styblinski_tang':
        return StyblinskiTang()
    elif bb_name == 'schwefel':
        return Schwefel()

    # constrained spaces
    elif bb_name == 'g1':
        return G1()
    elif bb_name == 'g3':
        return G3()
    elif bb_name == 'g4':
        return G4()
    elif bb_name == 'g6':
        return G6()
    elif bb_name == 'g7':
        return G7()
    elif bb_name == 'g10':
        return G10()
    elif bb_name == 'alkylation':
        return Alkylation()

    # mixed variable spaces
    elif bb_name == 'pressure_vessel':
        return PressureVessel()
    elif bb_name == 'vae_nas':
        return VAESmall()
    else:
        raise ValueError(f"'{bb_name}' is not a valid 'bb_name'!")

def preprocess_data(call_func):
    def _preprocess_data(self, x, *args, **kwargs):
        # inverse trafo the inputs if one-hot encoding is active
        if issubclass(type(self), CatSynFunc):
            x = self.inv_trafo_inputs(x)

        # round all integer features to the next integer
        self.round_integers(x)

        # query the black-box function
        f = call_func(self, x, *args, **kwargs)
        return f
    return _preprocess_data

class SynFunc:
    """base class for synthetic benchmark functions for which the optimum is known."""
    def __init__(self):
        # define index sets for categorical and integer variables
        self.cat_idx = set()
        self.int_idx = set()

        # define empty lists for inequality and equality constraints
        self.ineq_constr_funcs = []
        self.eq_constr_funcs = []

        # define if function is nonconvex
        self.is_nonconvex = False

    def round_integers(self, x):
        # rounds all integer features to integers
        #   this function assumes the 'non hot-encoded' state of the x_vals
        for idx in range(len(x)):
            if idx in self.int_idx:
                x[idx] = round(x[idx])

    def get_space(self):
        from .space import Space
        return Space(self.get_bounds(), int_idx=self.int_idx)

    def get_skopt_space(self):
        from skopt.space.space import Space as SkoptSpace
        from skopt.space.space import Categorical, Integer, Real

        skopt_bnds = []
        for idx, d in enumerate(self.get_bounds()):
            if idx in self.cat_idx:
                skopt_bnds.append(Categorical(d, transform='onehot'))
            elif idx in self.int_idx:
                skopt_bnds.append(Integer(low=int(d[0]), high=int(d[1])))
            else:
                skopt_bnds.append(Real(low=float(d[0]), high=float(d[1])))
        return SkoptSpace(skopt_bnds)

    def get_bounds(self):
        return []

    def get_lb(self):
        return [b[0] for b in self.get_bounds()]

    def get_ub(self):
        return [b[1] for b in self.get_bounds()]

    def get_model_core(self):
        if not self.has_constr():
            return None
        else:
            # define model core
            space = self.get_space()
            from .optimizer_utils import get_opt_core
            model_core = get_opt_core(space)

            # add equality constraints to model core
            for func in self.eq_constr_funcs:
                model_core.addConstr(func(model_core._cont_var_dict) == 0.0)

            # add inequality constraints to model core
            for func in self.ineq_constr_funcs:
                model_core.addConstr(func(model_core._cont_var_dict) <= 0.0)

            # set solver parameter if function is nonconvex
            model_core.Params.LogToConsole = 0
            if self.is_nonconvex:
                model_core.Params.NonConvex = 2

            model_core.update()
            return model_core

    def has_constr(self):
        return self.eq_constr_funcs or self.ineq_constr_funcs

    def get_num_constr(self):
        return len(self.eq_constr_funcs + self.ineq_constr_funcs)

    def is_feas(self, x):
        if not self.has_constr():
            return True

        # check if any constraint is above feasibility threshold
        for val in self.get_feas_vals(x):
            if val > 1e-5:
                return False
        return True

    def get_feas_vals(self, x):
        return self.get_feas_eq_vals(x) + self.get_feas_ineq_vals(x)

    def get_feas_eq_vals(self, x):
        # compute individual feasibility vals for all constr.
        if not self.eq_constr_funcs:
            return []
        return [func(x) for func in self.eq_constr_funcs]

    def get_feas_ineq_vals(self, x):
        # compute individual feasibility vals for all constr.
        if not self.ineq_constr_funcs:
            return []
        return [max(0, func(x)) for func in self.ineq_constr_funcs]

    def get_feas_penalty(self, x):
        # compute squared penalty of constr. violation vals
        if not self.has_constr():
            return 0.0

        feas_penalty = 0.0
        for vals in self.get_feas_vals(x):
            feas_penalty += vals**2
        return feas_penalty

    def get_init_data(self, num_init, rnd_seed, eval_constr=True):
        data = {'X': [], 'y': []}

        x_init = self.get_random_x(num_init, rnd_seed, eval_constr=eval_constr)

        for xi in x_init:
            data['X'].append(xi)
            data['y'].append(self(xi))

        return data

    def get_random_x(self, num_points, rnd_seed, eval_constr=True):
        # initial space
        temp_space = self.get_skopt_space()
        x_vals = []

        # generate rnd locations
        for xi in temp_space.rvs(num_points, random_state=rnd_seed):
            x_vals.append(xi)

        # return rnd locations
        if not self.has_constr() or not eval_constr:
            return x_vals

        # return constr projected rnd locations
        else:
            proj_x_vals = []

            for x in x_vals:
                # project init point into feasible region
                model_core = self.get_model_core()
                expr = [(xi - model_core._cont_var_dict[idx]) ** 2
                        for idx, xi in enumerate(x)]

                model_core.setObjective(expr=sum(expr))

                model_core.Params.LogToConsole = 0
                model_core.Params.TimeLimit = 5

                # add nonconvex parameters if constr make problem nonconvex
                if self.is_nonconvex:
                    model_core.Params.NonConvex = 2

                model_core.optimize()

                x_sol = [model_core._cont_var_dict[idx].x
                         for idx in range(len(self.get_bounds()))]
                proj_x_vals.append(x_sol)

            return proj_x_vals

class Himmelblau1D(SynFunc):
    def __call__(self, x, **kwargs):
        f = (x[0]**2 - 0.5) ** 2
        # output should be N(f_bar; 0, 1)
        f_bar = (f - 7/60) * np.sqrt(525/4)
        return f_bar * np.sin(10.0 * x[0])
    
    def get_bounds(self):
        return [[0.0, 1.0]]

class Hartmann6D(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py

    def __call__(self, x, **kwargs):
        a = np.asarray([
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])

        c = np.asarray([1.0, 1.2, 3.0, 3.2])
        p = np.asarray([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        s = 0

        for i in range(1, 5):
            sm = 0
            for j in range(1, 7):
                sm = sm + a[i - 1, j - 1] * (x[j - 1] - p[i - 1, j - 1]) ** 2
            s = s + c[i - 1] * np.exp(-sm)

        y = -s
        return y

    def get_bounds(self):
        return [[0.0, 1.0] for _ in range(6)]

class Rastrigin(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim

    def __call__(self, x, **kwargs):
        d = self.dim
        total = 0
        for xi in x:
            total = total + (xi ** 2 - 10.0 * np.cos(2.0 * np.pi * xi))
        f = 10.0 * d + total
        return f

    def get_bounds(self):
        return [[-4.0, 5.0] for _ in range(self.dim)]

class StyblinskiTang(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim

    def __call__(self, x, **kwargs):
        d = self.dim
        sum = 0
        for ii in range(1, d + 1):
            xi = x[ii - 1]
            new = xi ** 4 - 16 * xi ** 2 + 5 * xi
            sum = sum + new

        y = sum / 2.0
        return y

    def get_bounds(self):
        return [[-5.0, 5.0] for _ in range(self.dim)]

class Schwefel(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim

    def __call__(self, x, **kwargs):
        d = self.dim
        total = 0
        for ii in range(d):
            xi = x[ii]
            total = total + xi * np.sin(np.sqrt(abs(xi)))
        f = 418.9829 * d - total
        return f

    def get_bounds(self):
        return [[-500.0, 500.0] for _ in range(self.dim)]

class G1(SynFunc):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page506.htm
    def __init__(self):
        super().__init__()
        self.ineq_constr_funcs = [
            lambda x: 2*x[0] + 2*x[1] + x[9] + x[10] - 10,  #g1
            lambda x: 2*x[0] + 2*x[2] + x[9] + x[11] - 10,  #g2
            lambda x: 2*x[1] + 2*x[2] + x[10] + x[11] - 10, #g3
            lambda x: -8*x[0] + x[9],                       #g4
            lambda x: -8*x[1] + x[10],                      #g5
            lambda x: -3*x[2] + x[11],                      #g6
            lambda x: -2*x[3] - x[4] + x[9],                #g7
            lambda x: -2*x[5] - x[6] + x[10],               #g8
            lambda x: -2*x[7] - x[8] + x[11]                #g9
        ]

    def get_bounds(self):
        bnds = []
        for idx in range(13):
            lb = 0.0
            ub = 1.0 if idx not in (9, 10, 11) else 100.0
            bnds.append((lb, ub))
        return bnds

    def __call__(self, x, **kwargs):
        f = 5*sum(x[i] for i in range(4)) - \
            5*sum(x[i]**2 for i in range(4)) - \
            sum(x[i] for i in range(4, 13))
        return f

class G3(SynFunc):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2613.htm
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim
        self.is_nonconvex = True
        self.eq_constr_funcs = [
            lambda x: sum([x[i]*x[i] for i in range(self.dim)]) - 1  #h1
        ]

    def __call__(self, x, **kwargs):
        from math import sqrt
        f = (sqrt(self.dim)**self.dim)*np.prod([x[i] for i in range(self.dim)])
        f = -float(f)
        return f

    def get_bounds(self):
        return [(0.0, 1.0) for _ in range(self.dim)]

class G4(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        u = lambda x: 85.334407 + 0.0056858 * x[1] * x[4] + \
                      0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        v = lambda x: 80.51249 + 0.0071317 * x[1] * x[4] + \
                      0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2
        w = lambda x: 9.300961 + 0.0047026 * x[2] * x[4] + \
                      0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

        self.ineq_constr_funcs = [
            lambda x: -u(x),        #g1
            lambda x: u(x) - 92.0,  #g2
            lambda x: -v(x) + 90.0, #g3
            lambda x: v(x) - 110.0, #g4
            lambda x: -w(x) + 20.0, #g5
            lambda x: w(x) - 25.0,  #g6
        ]

    def get_bounds(self):
        lb = [78.0, 33.0, 27.0, 27.0, 27.0]
        ub = [102.0, 45.0, 45.0, 45.0, 45.0]
        return [(lb[idx], ub[idx]) for idx in range(5)]

    def __call__(self, x, **kwargs):
        f = 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
        return f

class G6(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        self.ineq_constr_funcs =[
            lambda x: - (x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100.0,
            lambda x: (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81
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
        self.ineq_constr_funcs =[
            lambda x: 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7] - 105,
            lambda x: 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7],
            lambda x: -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12,
            lambda x: 3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120,
            lambda x: 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40,
            lambda x: 0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5] - 30,
            lambda x: x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5],
            lambda x: -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]
        ]

    def get_bounds(self):
        return [(-10.0, 10.0) for _ in range(10)]

    def __call__(self, x, **kwargs):
        f = x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + \
            (x[2] - 10) ** 2 + 4 * (x[3] - 5) ** 2 + (x[4] - 3) ** 2 + \
            2 * (x[5] - 1) ** 2 + 5 * x[6] ** 2 + 7 * (x[7] - 11) ** 2 + \
            2 * (x[8] - 10) ** 2 + (x[9] - 7) ** 2 + 45
        return f

class G10(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        self.ineq_constr_funcs =[
            lambda x: -1 + 0.0025 * (x[3] + x[5]),
            lambda x: -1 + 0.0025 * (-x[3] + x[4] + x[6]),
            lambda x: -1 + 0.01 * (-x[4] + x[7]),
            lambda x: 100 * x[0] - x[0] * x[5] + 833.33252 * x[3] - 83333.333,
            lambda x: x[1] * x[3] - x[1] * x[6] - 1250 * x[3] + 1250 * x[4],
            lambda x: x[2] * x[4] - x[2] * x[7] - 2500 * x[4] + 1250000
        ]

    def get_bounds(self):
        lb = [100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        ub = [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        return [(lb[idx], ub[idx]) for idx in range(8)]

    def __call__(self, x, **kwargs):
        f = x[0]+x[1]+x[2]
        return f

class Alkylation(SynFunc):
    # original source: R. N. Sauer, A. R. Colville and C. W. Bunvick,
    #                  ‘Computer points the way to more profits’, Hydrocarbon Process.
    #                  Petrol. Refiner. 43,8492 (1964).
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py

    def __init__(self):
        super().__init__()
        self.is_nonconvex = True

        def X1(x): return x[0]
        def X2(x): return x[1]
        def X3(x): return x[2]
        def X4(x): return x[3]
        def X5(x): return x[4]
        def X6(x): return x[5]
        def X7(x): return x[6]
        def x5(x): return 1.22 * X4(x) - X1(x)
        def x6(x): return (98000.0 * X3(x)) / (X4(x) * X6(x) + 1000.0 * X3(x))
        def x8(x): return (X2(x) + x5(x)) / X1(x)

        self.ineq_constr_funcs =[
            lambda x: 0.99 * X4(x) - (X1(x) * (1.12 + 0.13167 * x8(x) - 0.00667 * x8(x) ** 2)),
            lambda x: (X1(x) * (1.12 + 0.13167 * x8(x) - 0.00667 * x8(x) ** 2)) - (100.0 / 99.0) * X4(x),
            lambda x: 0.99 * X5(x) - (86.35 + 1.098 * x8(x) - 0.038 * x8(x) ** 2 + 0.325 * (x6(x) - 89.0)),
            lambda x: (86.35 + 1.098 * x8(x) - 0.038 * x8(x) ** 2 + 0.325 * (x6(x) - 89.0)) - (100.0 / 99.0) * X5(x),
            lambda x: 0.9 * X6(x) - (35.82 - 0.222 * X7(x)),
            lambda x: (35.82 - 0.222 * X7(x)) - (10.0 / 9.0) * X6(x),
            lambda x: 0.99 * X7(x) - (-133 + 3 * X5(x)),
            lambda x: (-133 + 3.0 * X5(x)) - (100.0 / 99.0) * X7(x),
            lambda x: x5(x) - 2000,
            lambda x: -x5(x),
            lambda x: x6(x) - 93.0,
            lambda x: 85.0 - x6(x),
            lambda x: x8(x) - 12.0,
            lambda x: 3.0 - x8(x)
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
        from leaf_gp.optimizer_utils import get_opt_core
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

        squ_x8 = model_core.addVar(lb=3.0 ** 2, ub=12.0 ** 2)
        model_core.addConstr(squ_x8 == x8 * x8)

        # add other constrs
        model_core.addConstr(0.99 * x[3] - (x[0] * (1.12 + 0.13167 * x8 - 0.00667 * squ_x8))
                             <= 0.0)
        model_core.addConstr((x[0] * (1.12 + 0.13167 * x8 - 0.00667 * squ_x8)) -
                             (100.0 / 99.0) * x[3] <= 0.0)
        model_core.addConstr(0.99 * x[4] - (86.35 + 1.098 * x8 - 0.038 * squ_x8 +
                             0.325 * (x6 - 89.0)) <= 0.0)
        model_core.addConstr((86.35 + 1.098 * x8 - 0.038 * squ_x8 + 0.325 * (x6 - 89.0)) -
                             (100.0 / 99.0) * x[4] <= 0.0)
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

class PressureVessel(SynFunc):
    # adapted from: https://www.scielo.br/j/lajss/a/ZsdRkGWRVtDdHJP8WTDFFpB/?format=pdf&lang=en

    def __init__(self):
        super().__init__()
        self.int_idx = {0, 1}
        self.is_nonconvex = True

        def X0(x): return x[0] * 0.0625
        def X1(x): return x[1] * 0.0625

        self.ineq_constr_funcs = [
            lambda x: -X0(x) + 0.0193 * x[2],
            lambda x: -X1(x) + 0.00954 * x[3],
            lambda x: -np.pi * x[3] * x[2] ** 2 - (4/3) * np.pi * x[2] ** 3 + 1296000,
            # this constr. is in the reference but is not necessary
            # lambda x: x[3] - 240
        ]

    def get_model_core(self):
        # define model core
        space = self.get_space()
        from leaf_gp.optimizer_utils import get_opt_core
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
        lb2, ub2 = self.get_bounds()[2]
        x2_squ = model_core.addVar(lb=lb2 ** 2, ub=ub2 ** 2)
        model_core.addConstr(x2_squ == x[2] * x[2])

        model_core.addConstr(-np.pi * x[3] * x2_squ - (4/3) * np.pi * x[2] * x2_squ + 1296000 <= 0)

        # this constr. is in the reference but is not necessary given the bounds
        # model_core.addConstr(x[3] - 240 <= 0)

        # set solver parameter if function is nonconvex
        model_core.Params.LogToConsole = 0
        if self.is_nonconvex:
            model_core.Params.NonConvex = 2

        model_core.update()

        return model_core

    def get_bounds(self):
        return [(1, 99), (1, 99), (10.0, 200.0), (10.0, 200.0)]

    @preprocess_data
    def __call__(self, x, **kwargs):
        # true vars X0 and X1 are integer multiples of 0.0625
        def X0(x): return x[0] * 0.0625
        def X1(x): return x[1] * 0.0625

        f = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * X1(x) * x[2] ** 2 + \
            3.1661 * x[3] * X0(x) ** 2 + 19.84 * x[2] * X0(x) ** 2
        return f

class CatSynFunc(SynFunc):
    """class for synthetic benchmark functions for which the optimum is known that have
    one or more categorical vars."""

    def __init__(self):
        super().__init__()
        self.bnds = []
        self._has_onehot_trafo = False
        self._has_label_trafo = False

    def has_onehot_trafo(self):
        return self._has_onehot_trafo

    def has_label_trafo(self):
        return self._has_label_trafo

    def get_onehot_idx(self, get_idx):
        # outputs the onehot idx for categorical var 'get_idx'

        curr_idx = 0
        for idx, b in enumerate(self.bnds):
            if idx == get_idx:
                if idx in self.cat_idx:
                    return set(range(curr_idx, curr_idx+len(b)))
                else:
                    return curr_idx
            if idx in self.cat_idx:
                curr_idx += len(b)
            else:
                curr_idx += 1

    def eval_onehot(self):
        if self.cat_idx:
            # transform categorical vars to 'onehot'
            self._has_label_trafo = False
            self._has_onehot_trafo = True

            # define bounds to make them compatible with skopt
            self.cat_trafo = self.get_skopt_space()

    def eval_label(self):
        if self.cat_idx:
            # transform categorical vars to 'label'
            self._has_label_trafo = True
            self._has_onehot_trafo = False

            # do a label trafo, i.e. assumes that all categories are unique
            self._label_map = {}
            self._inv_label_map = {}

            # _label_map and _inv_label_map store the integer to categorical mapping
            for feat_idx in self.cat_idx:
                feat_map = {cat: i for i, cat in enumerate(self.bnds[feat_idx])}
                self._label_map[feat_idx] = feat_map

                inv_feat_map = {i: cat for i, cat in enumerate(self.bnds[feat_idx])}
                self._inv_label_map[feat_idx] = inv_feat_map

    def eval_normal(self):
        # switches evaluation back to normal
        self._has_onehot_trafo = False
        self._has_label_trafo = False

    def inv_trafo_inputs(self, x):
        if self._has_onehot_trafo:
            return conv2list(self.cat_trafo.inverse_transform([x])[0])

        elif self._has_label_trafo:
            # return inverse trafe for labels
            inv_trafo_x = []
            for idx, xi in enumerate(x):
                inv_trafo_x.append(self._inv_label_map[idx][xi]
                                   if idx in self.cat_idx else xi)
            return conv2list(inv_trafo_x)

        else:
            return conv2list(x)

    def trafo_inputs(self, x):
        if self._has_onehot_trafo:
            return conv2list(self.cat_trafo.transform([x])[0])

        elif self._has_label_trafo:
            # return inverse trafe for labels
            trafo_x = []
            for idx, xi in enumerate(x):
                trafo_x.append(self._label_map[idx][xi]
                               if idx in self.cat_idx else xi)
            return conv2list(trafo_x)

        else:
            return conv2list(x)

    def get_space(self):
        from leaf_gp.model_utils import Space
        if self._has_onehot_trafo:
            return Space(self.get_bounds(), int_idx=self.int_idx)
        else:
            return Space(self.get_bounds(), int_idx=self.int_idx, cat_idx=self.cat_idx)

    def get_skopt_space(self):
        from skopt.space.space import Space as SkoptSpace
        from skopt.space.space import Categorical

        skopt_bnds = []
        for idx, d in enumerate(self.bnds):
            skopt_bnds.append(Categorical(d, transform='onehot')
                              if idx in self.cat_idx else d)
        return SkoptSpace(skopt_bnds)

    def get_bounds(self):
        if self._has_onehot_trafo:
            return self.cat_trafo.transformed_bounds

        elif self._has_label_trafo:
            trafo_bnds = []
            for idx, b in enumerate(self.bnds):
                trafo_bnds.append(tuple(sorted(self._inv_label_map[idx].keys()))
                                  if idx in self.cat_idx else b)
            return trafo_bnds
        else:
            return self.bnds

    def get_random_x(self, num_points, rnd_seed, eval_constr=True):
        # initial space
        temp_space = self.get_skopt_space()
        x_vals = []

        # generate rnd locations
        for xi in temp_space.rvs(num_points, random_state=rnd_seed):
            x_vals.append(xi)

        # return rnd locations
        if not self.has_constr() or not eval_constr:
            x_vals = [self.trafo_inputs(x) for x in x_vals]
            return x_vals

        # return constr projected rnd locations
        else:
            # saving curr_trafo_state and set to eval_label()
            curr_trafo_state = (self._has_onehot_trafo, self._has_label_trafo)
            self.eval_label()

            proj_x_vals = []

            for x in x_vals:
                # project init point into feasible region
                #   special case for categorical variables
                x_trafo = self.trafo_inputs(x)

                model_core = self.get_model_core()
                expr = []

                for idx in range(len(x)):
                    if idx in self.cat_idx:
                        expr.append(sum([model_core._cat_var_dict[idx][cat]
                                         for cat in model_core._cat_var_dict[idx]
                                         if cat != x_trafo[idx]]))
                    else:
                        expr.append((x_trafo[idx] - model_core._cont_var_dict[idx]) ** 2)

                model_core.setObjective(expr=sum(expr))

                model_core.Params.LogToConsole = 0
                model_core.Params.TimeLimit = 5

                # add nonconvex parameters if constr make problem nonconvex
                if self.is_nonconvex:
                    model_core.Params.NonConvex = 2

                model_core.optimize()

                from leaf_gp.optimizer_utils import get_opt_sol
                x_sol = get_opt_sol(self.get_space(), model_core)
                proj_x_vals.append(self.inv_trafo_inputs(x_sol))

            # recover curr_trafo_state
            self._has_onehot_trafo, self._has_label_trafo = curr_trafo_state

            proj_x_vals = [self.trafo_inputs(x) for x in proj_x_vals]

            return proj_x_vals

class VAESmall(CatSynFunc):
    # adapted from: https://arxiv.org/pdf/1907.01329.pdf
    # and: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

    def __init__(self):
        super().__init__()
        # keep track of y for pe
        self.y = []

        self.int_idx = {idx for idx in range(1, 21)}
        self.cat_idx = {idx for idx in range(21, 32)}
        # self.cat_idx = {idx for idx in range(21, 25)}#32)}

        self.is_hier = True # change later

        self._num_enc = self._num_dec = self._num_fc = 2

        self.ineq_constr_funcs = []

        # var keys to feat index
        self._var_keys  = [('learning_rate', (-4.0, -2.0)), # 0
                           # encoder layers
                           ('num_enc', (0,2)),
                           ('enc_l1_out_channel_size', (2,5)),
                           ('enc_l1_stride', (1,2)),
                           ('enc_l1_padding', (0,3)),  # 4

                           ('enc_l2_out_channel_size', (3,6)), # 5
                           ('enc_l2_stride', (1,2)),
                           ('enc_l2_padding', (0,3)), # 7

                           # fully-connected layer
                           ('num_fc_enc', (0, 2)), # 8
                           ('fc1_enc_size', (0, 15)),
                           ('latent_space_size', (16, 64)),
                           ('num_fc_dec', (0, 2)), # 11

                           # decoder layers
                           ('dec_input', (3, 6)),  # 12
                           ('num_dec', (0,2)),
                           ('dec_l1_stride', (1,2)),
                           ('dec_l1_padding', (0,3)),
                           ('dec_l1_out_padding', (0,1)), # 16

                           ('dec_l2_in_channel_size', (2,5)), # 17
                           ('dec_l2_stride', (1,2)),
                           ('dec_l2_padding', (0,3)),
                           ('dec_l2_out_padding', (0,1)), # 20

                           # categorical vars
                           ('enc_l1_kernel_size', (3, 5)), # 21
                           ('enc_l2_kernel_size', (3, 5)),
                           ('dec_l1_kernel_size', (3, 5)),
                           ('dec_l2_kernel_size', (3, 5)), # 24

                           ('enc_l1_act', ('relu', 'prelu', 'leaky_relu')), # 25
                           ('enc_l2_act', ('relu', 'prelu', 'leaky_relu')), # 26

                           ('fc_enc_l1_act', ('relu', 'prelu', 'leaky_relu')), # 27
                           ('fc_enc_l2_act', ('relu', 'prelu', 'leaky_relu')),
                           ('fc_dec_l1_act', ('relu', 'prelu', 'leaky_relu')),
                           ('fc_dec_l2_act', ('relu', 'prelu', 'leaky_relu')), # 30

                           ('dec_l1_act', ('relu', 'prelu', 'leaky_relu')), # 31
                           ]
        self._default_vals = {}

        self.bnds = [bnd for key, bnd in self._var_keys]

        from leaf_gp.vae_nas_utils import get_test_loss
        self._func = get_test_loss

    def is_feas(self, x):
        return True

    def get_feas_penalty(self, x):
        return 0.0

    def has_constr(self):
        return True

    def _get_var_map(self):
        # define var_map
        var_map = {}
        for idx, var_tuple in enumerate(self._var_keys):
            key, bnd = var_tuple
            var_map[key] = idx
        return var_map

    def _get_base_constr_model(self):
        # define model core
        space = self.get_space()
        from leaf_gp.optimizer_utils import get_opt_core
        model_core = get_opt_core(space)

        # add helper vars
        x_con = model_core._cont_var_dict
        x_cat = model_core._cat_var_dict
        var_map = self._get_var_map()

        def add_full_convo_layer(model, input, layer_idx, var_map):
            # get conv params
            # kernel is a categorical choice
            k_idx = var_map[f'enc_l{layer_idx}_kernel_size']
            k = model.addVar(
                name=f"enc_l{layer_idx}_kernel_size", vtype="I")
            model.addConstr(k == 3*x_cat[k_idx][0] + 5*x_cat[k_idx][1])

            s = x_con[var_map[f'enc_l{layer_idx}_stride']]
            p = x_con[var_map[f'enc_l{layer_idx}_padding']]

            # add constr for conv output size
            conv_out = model.addVar(
                name=f"conv_out_{layer_idx}", vtype="I")

            model.addConstr(
                s * conv_out == input - k + 2 * p + s)
            return conv_out

        def add_full_deconvo_layer(model, input, layer_idx, var_map):
            # get conv params
            # kernel is a categorical choice
            k_idx = var_map[f'dec_l{layer_idx}_kernel_size']
            k = model.addVar(
                name=f"dec_l{layer_idx}_kernel_size", vtype="I")
            model.addConstr(k == 3*x_cat[k_idx][0] + 5*x_cat[k_idx][1])

            s = x_con[var_map[f'dec_l{layer_idx}_stride']]
            p = x_con[var_map[f'dec_l{layer_idx}_padding']]
            o = x_con[var_map[f'dec_l{layer_idx}_out_padding']]

            # output_padding needs to be smaller than stride or dilation
            # see pytorch docs
            model.addConstr(o + 1 <= s)

            # add constr for conv output size
            deconv_out = model.addVar(
                name=f"deconv_out_{layer_idx}", vtype="I")

            model.addConstr(
                deconv_out == (input - 1)*s + k - 2 * p + o)
            return deconv_out

        ### define encoder layers
        curr_input = 28

        # add bin vars to indicate enc layers are active
        enc_act = [model_core.addVar(name=f"enc_layer_act_{layer_idx}", vtype="B")
                   for layer_idx in range(1, self._num_enc + 1)]

        model_core.addConstr(sum(enc_act) == x_con[var_map['num_enc']])

        for layer_idx in range(1, self._num_enc):
            model_core.addConstr(enc_act[layer_idx - 1] >= enc_act[layer_idx])

        # define layer output conditions
        for layer_idx in range(1, self._num_enc + 1):
            conv_out = add_full_convo_layer(
                    model_core, curr_input, layer_idx, var_map)

            # compute layer out depending on whether it's active or not
            layer_out = model_core.addVar(
                name=f"enc_layer_out_{layer_idx}", vtype="I")

            layer_act = enc_act[layer_idx - 1]

            model_core.addConstr(
                layer_out == layer_act * conv_out + (1 - layer_act) * curr_input)

            curr_input = layer_out

        model_core.addConstr(curr_input >= 1)

        ### define fc layers
        fc_enc_act = [model_core.addVar(name=f"fc_layer_act_{layer_idx}", vtype="B")
                      for layer_idx in range(1, self._num_fc + 1)]

        model_core.addConstr(sum(fc_enc_act) == x_con[var_map['num_fc_enc']])

        for layer_idx in range(1, self._num_fc):
            model_core.addConstr(fc_enc_act[layer_idx - 1] >= fc_enc_act[layer_idx])

        fc_dec_act = [model_core.addVar(name=f"dec_layer_act_{layer_idx}", vtype="B")
                      for layer_idx in range(1, self._num_fc + 1)]

        model_core.addConstr(sum(fc_dec_act) == x_con[var_map['num_fc_dec']])

        for layer_idx in range(1, self._num_fc):
            model_core.addConstr(fc_dec_act[layer_idx - 1] >= fc_dec_act[layer_idx])

        ### define decoder layers

        # add bin vars to indicate enc layers are active
        dec_act = [model_core.addVar(name=f"dec_layer_act_{layer_idx}", vtype="B")
                   for layer_idx in range(1, self._num_dec + 1)]

        model_core.addConstr(sum(dec_act) == x_con[var_map['num_dec']])

        for layer_idx in range(1, self._num_dec):
            model_core.addConstr(dec_act[layer_idx - 1] >= dec_act[layer_idx])

        curr_input = 7
        for layer_idx in range(1, self._num_dec + 1):
            deconv_out = add_full_deconvo_layer(
                model_core, curr_input, layer_idx, var_map)

            # compute layer out depending on whether it's active or not
            layer_out = model_core.addVar(
                name=f"dec_layer_out_{layer_idx}", vtype="I")

            layer_act = dec_act[layer_idx - 1]

            model_core.addConstr(
                layer_out == layer_act * deconv_out + (1 - layer_act) * curr_input)

            curr_input = layer_out

        # two outcomes according to paper:
        # 1. output should be 28 if deconvolutions are active
        # 2. output should be params['dec_input'] * 7 * 7 = 28 * 28 if no deconv is active
        model_core.addConstr(
            (dec_act[0] == 1) >> (curr_input == 28)
        )
        model_core.addConstr(
            (dec_act[0] == 0) >> (x_con[var_map['dec_input']] == 4)
        )

        # REMOVE later
        # model_core.addConstr(x_con[var_map['num_dec']] == 0)

        self._enc_act = enc_act
        self._fc_enc_act = fc_enc_act
        self._fc_dec_act = fc_dec_act
        self._dec_act = dec_act

        model_core.update()
        return model_core

    def get_model_core(self, size_is_cat=False):
        model_core = self._get_base_constr_model()
        model_core.Params.LogToConsole = 0
        model_core.Params.NonConvex = 2

        # add hierarchical constr
        x_con = model_core._cont_var_dict
        x_cat = model_core._cat_var_dict
        var_map = self._get_var_map()

        # set default values for inactive encoder layers
        for layer_idx in range(1, self._num_enc + 1):

            lb = self._var_keys[var_map[f'enc_l{layer_idx}_out_channel_size']][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'enc_l{layer_idx}_out_channel_size']] <= lb)
            )

            lb = self._var_keys[var_map[f'enc_l{layer_idx}_stride']][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'enc_l{layer_idx}_stride']] <= lb)
            )

            lb = self._var_keys[var_map[f'enc_l{layer_idx}_padding']][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'enc_l{layer_idx}_padding']] <= lb)
            )

            ## add categorical constraints
            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0) >> \
                (x_cat[var_map[f'enc_l{layer_idx}_kernel_size']][0] == 1)
            )

            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0) >> \
                (x_cat[var_map[f'enc_l{layer_idx}_act']][0] == 1)
            )

        # set default values for inactive fc layers
        for layer_idx in range(1, self._num_fc + 1):
            if layer_idx == 1:
                # number of nodes are only set for first enc / dec layers
                lb = self._var_keys[var_map[f'fc1_enc_size']][1][0]
                model_core.addConstr(
                    (self._fc_enc_act[layer_idx-1] == 0) >> \
                    (x_con[var_map[f'fc1_enc_size']] <= lb)
                )

            ## add categorical constraints
            model_core.addConstr(
                (self._fc_enc_act[layer_idx - 1] == 0) >> \
                (x_cat[var_map[f'fc_enc_l{layer_idx}_act']][0] == 1)
            )

            model_core.addConstr(
                (self._fc_dec_act[layer_idx - 1] == 0) >> \
                (x_cat[var_map[f'fc_dec_l{layer_idx}_act']][0] == 1)
            )

        # set default values for inactive decoder layers
        for layer_idx in range(1, self._num_dec + 1):

            if layer_idx == 2:
                # input channels only relevant if layer 2
                lb = self._var_keys[var_map[f'dec_l{layer_idx}_in_channel_size']][1][0]
                model_core.addConstr(
                    (self._dec_act[layer_idx-1] == 0) >> \
                    (x_con[var_map[f'dec_l{layer_idx}_in_channel_size']] <= lb)
                )

            lb = self._var_keys[var_map[f'dec_l{layer_idx}_stride']][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'dec_l{layer_idx}_stride']] <= lb)
            )

            lb = self._var_keys[var_map[f'dec_l{layer_idx}_padding']][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'dec_l{layer_idx}_padding']] <= lb)
            )

            lb = self._var_keys[var_map[f'dec_l{layer_idx}_out_padding']][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx-1] == 0) >> \
                (x_con[var_map[f'dec_l{layer_idx}_out_padding']] <= lb)
            )

            ## add categorical constraints
            model_core.addConstr(
                (self._dec_act[layer_idx - 1] == 0) >> \
                (x_cat[var_map[f'dec_l{layer_idx}_kernel_size']][0] == 1)
            )

            if layer_idx == 1:
                model_core.addConstr(
                    (self._dec_act[layer_idx - 1] == 0) >> \
                    (x_cat[var_map[f'dec_l{layer_idx}_act']][0] == 1)
                )

        model_core.Params.NonConvex = 2
        model_core.update()

        return model_core

    @preprocess_data
    def __call__(self, x, **kwargs):
        temp_dict = dict()

        for idx, key_tuple in enumerate(self._var_keys):
            key, bnd = key_tuple

            if key == 'learning_rate':
                temp_dict[key] = 10.0 ** x[idx]
            elif key == 'fc1_enc_size':
                temp_dict[key] = 64 * x[idx]
            elif key.split('_')[-2] == 'channel' or key == 'dec_input':
                temp_dict[key] = 2 ** x[idx]
            else:
                temp_dict[key] = x[idx]

        temp_dict.update(self._default_vals)

        try:
            f = self._func(temp_dict)
        except:
            f = max(self.y) if self.y else 500

        self.y.append(f)
        return f