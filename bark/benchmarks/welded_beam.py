import numpy as np
import pandas as pd
from math import sqrt
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs, Constraints
from bofire.data_models.features.api import (
    ContinuousInput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
)
from bark.bofire_utils.constraints import FunctionalInequalityConstraint
from bark.bofire_utils.domain import build_integer_input 


class WeldedBeam(Benchmark):
    """
    Welded Beam problem from https://link.springer.com/content/pdf/10.1007/s00158-018-2182-1.pdf

    Only implementing for continuous inputs.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Constants
        self.F = 6000
        self.delta_max = 0.25
        self.L = 14
        
        # Material parameters (C1, C2, sigma_d, E, G)
        self.material_params = {
            0: (0.1047, 0.0481, 3e4, 3e7, 12e6),  # steel
            1: (0.0489, 0.0224, 8e3, 14e6, 6e6),  # cast iron
            2: (0.5235, 0.2405, 5e3, 1e7, 4e6),   # aluminum
            3: (0.5584, 0.2566, 8e3, 16e6, 1e7),  # brass
        }
        
        # Define domain
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    build_integer_input(key="w", bounds=[0, 1]),  # binary, welding type
                    CategoricalInput(key="m", categories=[0, 1, 2, 3]),  # material type
                    ContinuousInput(key="h", bounds=(0.0625, 2)),
                    ContinuousInput(key="l", bounds=(0.1, 10)),
                    ContinuousInput(key="t", bounds=(2, 20)),
                    ContinuousInput(key="b", bounds=(0.0625, 2)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="manufacturing_cost", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    FunctionalInequalityConstraint(
                        func=self._constraint_g1,
                        rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=self._constraint_g2,
                        rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=self._constraint_g3,
                        rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=self._constraint_g4,
                        rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=self._constraint_g5,
                        rhs=0.0
                    ),
                ]
            ),
        )

    def unpack_inputs(self, x):
        vars = {k: x[i] for i, k in enumerate(self.domain.inputs.get_keys())}
        return vars["w"], vars["m"], vars["h"], vars["l"], vars["t"], vars["b"]

    def _get_beam_params(self, x):
        """Calculate beam parameters for a single configuration"""
        w, m, h, l, t, b = self.unpack_inputs(x)
        
        if w == 0:
            # TODO: think if we want to use sqrt from corelib or sth else?
            A = sqrt(2) * h * l
            J = A * ((h + t)**2 / 4 + l**2 / 12)
            R = 0.5 * sqrt(l**2 + (h + t)**2)
        else:
            A = sqrt(2) * h * (t + l)
            J = (
                sqrt(2) * h * l * ((h + t)**2 / 4 + l**2 / 12) + 
                sqrt(2) * h * t * ((h + l)**2 / 4 + t**2 / 12)
            )
            R = 0.5 * max(
                sqrt(l**2 + (h + t)**2), 
                sqrt(t**2 + (h + l)**2)
            )
            
        cos_theta = l / (2 * R)
        return A, J, R, cos_theta

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate objective value (cost)"""
        x = X[self.domain.inputs.get_keys()].to_numpy()
        
        costs = []
        for xi in x:
            w, m, h, l, t, b = self.unpack_inputs(xi)
            C1, C2, _, _, _ = self.material_params[int(m)]
            cost = (1 + C1) * (w * t + l) * h**2 + C2 * t * b * (self.L + l)
            costs.append(cost)
            
        return pd.DataFrame(data=np.array(costs).reshape(-1, 1), 
                          columns=self.domain.outputs.get_keys())

    def _constraint_g1(self, x, model_core=None):
        """Shear stress constraint"""
        w, m, h, l, t, b = self.unpack_inputs(x)
        A, J, R, cos_theta = self._get_beam_params(x)
        _, _, sigma_d, _, _ = self.material_params[int(m)]
        
        tau_prime = sqrt(self.F / A)
        tau_double_prime = self.F * (self.L + 0.5 * l) * R / J
        tau = sqrt(tau_prime**2 + tau_double_prime**2 + 
                  2 * tau_prime * tau_double_prime * cos_theta)
        return tau - 0.577 * sigma_d

    def _constraint_g2(self, x, model_core=None):
        """Normal stress constraint"""
        w, m, h, l, t, b = self.unpack_inputs(x)
        _, _, sigma_d, _, _ = self.material_params[int(m)]
        
        sigma = 6 * self.F * self.L / (t**2 * b)
        return sigma - sigma_d

    def _constraint_g3(self, x, model_core=None):
        """Width constraint"""
        _, _, h, _, _, b = self.unpack_inputs(x)
        return h - b

    def _constraint_g4(self, x, model_core=None):
        """Buckling load constraint"""
        w, m, h, l, t, b = self.unpack_inputs(x)
        _, _, _, E, G = self.material_params[int(m)]
        
        P_c = (4.013 * t * b**3 * sqrt(E * G) / (6 * self.L**2) * 
               (1 - t / (4 * self.L) * sqrt(E / G)))
        return self.F - P_c

    def _constraint_g5(self, x, model_core=None):
        """Deflection constraint"""
        w, m, h, l, t, b = self.unpack_inputs(x)
        _, _, _, E, _ = self.material_params[int(m)]
        
        delta = 4 * self.F * self.L**3 / (E * t**3 * b)
        return delta - self.delta_max