#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import unittest

import numpy as np
import pandas as pd
from src.pyTensileTest import tensile_test_data


class TestTensileTest(unittest.TestCase):

    def test_tensile_test(self):

        x = np.linspace(0, 200, 1000)
        y = 3 * x

        test_data = [pd.DataFrame(np.array([x, y]).T, columns=['strain', 'stress'])]

        tens_test = tensile_test_data.tensile_test(
            test_data, 'DataFrame', unit_strain='')

        tens_test.calc_e_modulus()
        tens_test.calc_strength()
        tens_test.calc_toughness()
        tens_test.calc_elongation_at_break()
        tens_test.generate_plots()

        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.e_modulus_title], 3, 5)
        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.strength_title], y.max(), 5)
        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.elongation_at_break_title],
            x.max(), 5)

        tens_test_2 = tensile_test_data.tensile_test(
            'tests/tensile_test.xlsx', 'import', unit_strain='%',
            offset_correction=True)
        tens_test_2.find_data_borders(
            onset_mode='stress_thresh', stress_thresh=1,
            data_end_mode='lower_thresh', lower_thresh=-3)
        tens_test_2.find_data_borders(
            onset_mode='deriv_2_max', stress_thresh=1,
            data_end_mode='perc_drop', lower_thresh=0.05, drop_window=2)
        tens_test_2.find_data_borders(
            onset_mode='fit', fit_function='cum_dist_normal_with_rise',
            fit_boundaries=[[0.01, 5], [0, 12], [0, 12], [0, 5]],
            data_end_mode='perc_drop', lower_thresh=0.05, drop_window=2)
