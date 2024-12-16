#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple, Union

import numpy as np


"""
Problem formulation for the RF Coverage and Capacity Optimization (CCO) problem.
"""


class CCORasterBlanketFormulation:
    """Generate combined reward, over all raster locations, from dual objectives
    of minimizing under coverage (holes) and over coverage (interference).

    RSRP: Reference Signal Receive Power
    Coverage holes:  Z = h(x) = weak_coverage_threshold - RSRP_From_TargetCell(x),
                        where x is the location.
    Over coverage:   Y = g(x) = sum(RSRP_From_NeighborCells(x))
                                + over_coverage_threshold
                                - RSRP_From_TargetCell(x),
                        where RSRP_From_TargetCell > weak_coverage_threshold

    Suggested : weak_coverage_threshold = -90 dBm, over_coverage_threshold = 6 dBm

    Metrics: percentages of coverage holes and over coverage
    """

    def __init__(
        self,
        weak_coverage_threshold: float = -80,
        over_coverage_threshold: float = 6,
    ):
        self.weak_coverage_threshold = weak_coverage_threshold
        self.over_coverage_threshold = over_coverage_threshold


    def get_reward_components(
        self, rsrp_map: np.ndarray, interference_map: np.ndarray
    ) -> Tuple[float, float]:
        """Get individual reward components from all the locations in the map"""
        weak_coverage_area, over_coverage_area = self.get_weak_over_coverage_area(
            rsrp_map,
            interference_map,
            self.weak_coverage_threshold,
            self.over_coverage_threshold,
        )

        f_weak_coverage = CCORasterBlanketFormulation.activation_function(
            self.weak_coverage_threshold - rsrp_map[weak_coverage_area]
        ).sum()

        g_over_coverage = CCORasterBlanketFormulation.activation_function(
            interference_map[over_coverage_area]
            + self.over_coverage_threshold
            - rsrp_map[over_coverage_area]
        ).sum()

        return f_weak_coverage, g_over_coverage

    def get_weak_over_coverage_area_percentages(
        self, rsrp_map: np.ndarray, interference_map: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate the percentages of coverage hole and over coverage area"""
        size = rsrp_map.size
        weak_coverage_area, over_coverage_area = self.get_weak_over_coverage_area(
            rsrp_map,
            interference_map,
            self.weak_coverage_threshold,
            self.over_coverage_threshold,
        )
        weak_coverage_percentage = weak_coverage_area.sum() / size
        over_coverage_percentage = over_coverage_area.sum() / size
        return weak_coverage_percentage, over_coverage_percentage

    @staticmethod
    def get_weak_over_coverage_area(
        rsrp_map: np.ndarray,
        interference_map: np.ndarray,
        weak_coverage_threshold: float,
        over_coverage_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the weak coverage and over coverage areas
        as 2D boolean indicator matrices.
        """
        weak_coverage_area = rsrp_map < weak_coverage_threshold
        over_coverage_area = (rsrp_map >= weak_coverage_threshold) & (
            interference_map + over_coverage_threshold > rsrp_map
        )
        return weak_coverage_area, over_coverage_area

    @staticmethod
    def activation_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Sigmoid Function"""
        return 1 / (1 + np.exp(-x))