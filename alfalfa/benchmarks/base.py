import abc

import numpy as np
import skopt.space.space as skopt_space
import torch
from jaxtyping import Int, Shaped

from ..optimizer.optimizer_utils import conv2list, get_opt_core, get_opt_sol
from ..utils.space import Space


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

    is_nonconvex = False
    is_vectorised = False

    def __init__(self):
        # define index sets for categorical and integer variables
        self.cat_idx = set()
        self.int_idx = set()

        # define empty lists for inequality and equality constraints
        self.ineq_constr_funcs = []
        self.eq_constr_funcs = []

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def get_bounds(self):
        pass

    def round_integers(self, x):
        # rounds all integer features to integers
        #   this function assumes the 'non hot-encoded' state of the x_vals
        for idx in range(len(x)):
            if idx in self.int_idx:
                x[idx] = round(x[idx])

    def get_space(self):
        return Space(self.get_bounds(), int_idx=self.int_idx)

    def get_skopt_space(self):
        skopt_bnds = []
        for idx, d in enumerate(self.get_bounds()):
            if idx in self.cat_idx:
                skopt_bnds.append(skopt_space.Categorical(d, transform="onehot"))
            elif idx in self.int_idx:
                skopt_bnds.append(skopt_space.Integer(low=int(d[0]), high=int(d[1])))
            else:
                skopt_bnds.append(skopt_space.Real(low=float(d[0]), high=float(d[1])))
        return skopt_space.Space(skopt_bnds)

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

    def get_init_data(self, num_init, rnd_seed, eval_constr=True, **kwargs):
        x_init = self.get_random_x(num_init, rnd_seed, eval_constr=eval_constr)

        xs = np.asarray(x_init)
        ys = self.vector_apply(xs, **kwargs)

        return (xs, ys)

    def grid_sample(
        self, shape: Int[np.ndarray, "D"]
    ) -> tuple[
        Shaped[np.ndarray, "{shape.prod()} D"], Shaped[np.ndarray, "{shape.prod()}"]  # type: ignore
    ]:
        """Return data sampled on a grid"""
        space = self.get_space()
        xs = [dim.grid_sample(s) for s, dim in zip(shape, space.dims)]
        test_x_mgrid = np.meshgrid(*xs, indexing="ij")
        flats = [x.flatten() for x in test_x_mgrid]
        test_x = np.stack(flats, axis=-1)
        test_y = self.vector_apply(test_x)

        return (test_x, test_y)

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
                expr = [
                    (xi - model_core._cont_var_dict[idx]) ** 2
                    for idx, xi in enumerate(x)
                ]

                model_core.setObjective(expr=sum(expr))

                model_core.Params.LogToConsole = 0
                model_core.Params.TimeLimit = 5

                # add nonconvex parameters if constr make problem nonconvex
                if self.is_nonconvex:
                    model_core.Params.NonConvex = 2

                model_core.optimize()

                x_sol = [
                    model_core._cont_var_dict[idx].x
                    for idx in range(len(self.get_bounds()))
                ]
                proj_x_vals.append(x_sol)

            return proj_x_vals

    def vector_apply(self, x, **kwargs):
        if self.is_vectorised:
            return self(x)

        ys = [self(xi) for xi in x]
        if isinstance(x, np.ndarray):
            return np.asarray(ys)
        elif isinstance(x, torch.Tensor):
            return torch.tensor(ys)
        else:
            return ys


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
                    return set(range(curr_idx, curr_idx + len(b)))
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
                inv_trafo_x.append(
                    self._inv_label_map[idx][xi] if idx in self.cat_idx else xi
                )
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
                trafo_x.append(self._label_map[idx][xi] if idx in self.cat_idx else xi)
            return conv2list(trafo_x)

        else:
            return conv2list(x)

    def get_space(self):
        if self._has_onehot_trafo:
            return Space(self.get_bounds(), int_idx=self.int_idx)
        else:
            return Space(self.get_bounds(), int_idx=self.int_idx, cat_idx=self.cat_idx)

    def get_skopt_space(self):
        skopt_bnds = []
        for idx, d in enumerate(self.bnds):
            skopt_bnds.append(
                skopt_space.Categorical(d, transform="onehot")
                if idx in self.cat_idx
                else d
            )
        return skopt_space.Space(skopt_bnds)

    def get_bounds(self):
        if self._has_onehot_trafo:
            return self.cat_trafo.transformed_bounds

        elif self._has_label_trafo:
            trafo_bnds = []
            for idx, b in enumerate(self.bnds):
                trafo_bnds.append(
                    tuple(sorted(self._inv_label_map[idx].keys()))
                    if idx in self.cat_idx
                    else b
                )
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
                        expr.append(
                            sum(
                                [
                                    model_core._cat_var_dict[idx][cat]
                                    for cat in model_core._cat_var_dict[idx]
                                    if cat != x_trafo[idx]
                                ]
                            )
                        )
                    else:
                        expr.append(
                            (x_trafo[idx] - model_core._cont_var_dict[idx]) ** 2
                        )

                model_core.setObjective(expr=sum(expr))

                model_core.Params.LogToConsole = 0
                model_core.Params.TimeLimit = 5

                # add nonconvex parameters if constr make problem nonconvex
                if self.is_nonconvex:
                    model_core.Params.NonConvex = 2

                model_core.optimize()

                x_sol = get_opt_sol(self.get_space(), model_core)
                proj_x_vals.append(self.inv_trafo_inputs(x_sol))

            # recover curr_trafo_state
            self._has_onehot_trafo, self._has_label_trafo = curr_trafo_state

            proj_x_vals = [self.trafo_inputs(x) for x in proj_x_vals]

            return proj_x_vals


class MFSynFunc(SynFunc):
    """Synthetic benchmarks with multiple fidelities."""

    @abc.abstractmethod
    def __call__(self, x, i):
        pass

    @property
    @abc.abstractmethod
    def costs(self):
        pass

    def get_init_data(
        self, fidelities: list[int], rnd_seed, eval_constr=True, **kwargs
    ):
        num_init = sum(fidelities)
        x_init = self.get_random_x(num_init, rnd_seed, eval_constr=eval_constr)
        i = np.repeat(np.arange(len(fidelities)), fidelities)
        xs = np.asarray(x_init)
        ys = self.vector_apply(xs, i, **kwargs)

        return (xs, i, ys)

    def vector_apply(self, x, i, **kwargs):
        if self.is_vectorised:
            return self(x, i)

        ys = [self(xi, ii) for xi, ii in zip(x, i)]
        if isinstance(x, np.ndarray):
            return np.asarray(ys)
        elif isinstance(x, torch.Tensor):
            return torch.tensor(ys)
        else:
            return ys
