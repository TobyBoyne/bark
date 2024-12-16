import logging
import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective

from bark.bofire_utils.domain import build_integer_input
from bark.benchmarks.cco.problem_formulation import CCORasterBlanketFormulation
from bark.benchmarks.cco.simulated_rsrp import SimulatedRSRP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CCOBench(Benchmark):
    """Coverage and Capacity optimization for Cell networks benchmark.
    
    Based on: R. M. Dreifuerst, et al. Optimizing Coverage and Capacity in Cellular 
    Networks using Machine Learning. IEEE ICASSP special session on Machine Learning 
    in Networks, 2021.

    Uses data from https://github.com/Ryandry1st/CCO-in-ORAN/tree/main
    """
    
    def __init__(
        self,
        n_int_values: int = 6,
        **kwargs
    ):
        """Initialize CCO benchmark.
        
        Args:
            n_int_values: Number of integer values for downtilt (6 or 11)
        """
        super().__init__(**kwargs)
        
        if n_int_values not in (6, 11):
            raise ValueError("Only 6 and 11 int values are supported")
            
        # Set up domain
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    # Downtilts (integers)
                    *[build_integer_input(
                        key=f"downtilt_{i}", 
                        bounds=[0, n_int_values-1]
                    ) for i in range(15)],
                    # Transmission powers (continuous)
                    *[ContinuousInput(
                        key=f"power_{i}",
                        bounds=(30.0, 50.0)
                    ) for i in range(15)]
                ]
            ),
            outputs=Outputs(
                features=[
                    ContinuousOutput(
                        key="balanced_objective",
                        objective=MinimizeObjective()
                    )
                ]
            )
        )
        
        # Load data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        powermaps_path = os.path.join(current_dir, "powermaps")
        data = {}
        for i in range(11):
            specific_matrix_path = os.path.join(powermaps_path, f"powermatrixDT{i}.npz")
            data[i] = dict(np.load(specific_matrix_path))
        self.data = data
        # Initialize CCO components
        self.n_int_values = n_int_values
        self.simulated_rsrp = SimulatedRSRP(
            powermaps=data,
            min_TX_power_dBm=30,
            max_TX_power_dBm=50,
        )
        self.problem_formulation = CCORasterBlanketFormulation()
        self.evaluations = 0
        
    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        """Evaluate CCO objectives for given configurations.
        
        Args:
            X: DataFrame with downtilt and power configurations
            
        Returns:
            DataFrame with objective values
        """
        logging.info(f"Evaluating {X.shape[0]} configurations on {self.evaluations}th evaluation")
        self.evaluations += 1
        # Convert to numpy array and reshape
        downtilt_keys = [key for key in self.domain.inputs.get_keys() if key.startswith("downtilt_")]
        power_keys = [key for key in self.domain.inputs.get_keys() if key.startswith("power_")]
        downtilts = X[downtilt_keys].to_numpy()
        tx_powers = X[power_keys].to_numpy()
        
        if self.n_int_values == 6:
            downtilts *= 2  # Scale integer values if using 6 values
            
        # Evaluate each configuration
        objectives = []
        for downtilts, tx_powers in zip(downtilts, tx_powers):
            # Get power maps
            rsrp_map, interference_map, _ = self.simulated_rsrp.get_RSRP_and_interference_powermap(
                downtilts=downtilts, tx_powers=tx_powers
            )
            
            # Calculate coverage metrics
            weak_coverage, over_coverage = self.problem_formulation.get_weak_over_coverage_area_percentages(
                rsrp_map=rsrp_map, interference_map=interference_map
            )
            
            objectives.append([0.5 * weak_coverage + 0.5 * over_coverage,])
                
        return pd.DataFrame(
            data=objectives,
            columns=self.domain.outputs.get_keys()
        )