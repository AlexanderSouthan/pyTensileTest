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

from little_helpers.math_functions import cum_dist_normal_with_rise

class TestTensileTest(unittest.TestCase):

    def test_tensile_test(self):

        x = np.linspace(0, 200, 1001)
        y = 3 * x

        test_data = [pd.DataFrame(np.array([x, y]).T,
                                  columns=['strain', 'stress'])]

        # test initializing with DataFrame input
        tens_test = tensile_test_data.tensile_test(
            test_data, 'DataFrame', unit_strain='')

        # test the different calculation methods
        tens_test.calc_e_modulus()
        tens_test.calc_strength()
        tens_test.calc_toughness()
        tens_test.calc_elongation_at_break()

        # test plotting method
        tens_test.generate_plots()

        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.e_modulus_title], 3, 5)
        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.strength_title], y.max(), 5)
        self.assertAlmostEqual(
            tens_test.results.loc[0, tens_test.elongation_at_break_title],
            x.max(), 5)

        # test initializing with Excel input
        tens_test_2 = tensile_test_data.tensile_test(
            'tests/tensile_test.xlsx', 'import', unit_strain='%',
            offset_correction=True, start_sheet=0, columns=[1, 2, 3],
            column_names=['strain', 'stress', 'tool_distance'], header_rows=1)

        # test the different methods for onset and data end detection
        tens_test_2.find_data_borders(
            onset_mode='stress_thresh', stress_thresh=1,
            data_end_mode='lower_thresh', lower_thresh=-3)
        tens_test_2.find_data_borders(
            onset_mode='deriv_2_max', stress_thresh=1,
            data_end_mode='perc_drop', lower_thresh=0.05, drop_window=2)
        tens_test_2.find_data_borders(
            onset_mode='fit', fit_function=cum_dist_normal_with_rise,
            fit_boundaries={'sigma': [0.01, 5], 'x_offset': [0, 12],
                            'slope': [0, 12], 'amp': [0, 5]},
            data_end_mode='perc_drop', lower_thresh=0.05, drop_window=2)

        # test with offset_strain window
        tens_test_3 = tensile_test_data.tensile_test(
            'tests/tensile_test.xlsx', 'import', unit_strain='%',
            onset_mode=None, data_end_mode=None, offset_correction=True,
            offset_strain=[0, 5], start_sheet=0, columns=[0, 1, 2],
            column_names=['strain', 'stress', 'tool_distance'], header_rows=1)

        # test the perc_drop data end detection
        tens_test_4 = tensile_test_data.tensile_test(
            'tests/tensile_test_2.xlsx', 'import', unit_strain='%',
            start_sheet=0, columns=[1, 2, 3],
            column_names=['strain', 'stress', 'tool_distance'], header_rows=1)
        tens_test_4.find_data_borders(
            data_end_mode='perc_drop', lower_thresh=0.01)
        self.assertAlmostEqual(
            tens_test_4.data_processed[0]['strain'].values[-1], 99.8, 5)

        # test the lower_thresh data end detection
        tens_test_5 = tensile_test_data.tensile_test(
            'tests/tensile_test_2.xlsx', 'import', unit_strain='%',
            start_sheet=0, columns=[1, 2, 3],
            column_names=['strain', 'stress', 'tool_distance'], header_rows=1,
            deriv_window=1)
        tens_test_5.find_data_borders(
            data_end_mode='lower_thresh', lower_thresh=-10)
        self.assertAlmostEqual( # expected value shifted by one due to deriv_window
            tens_test_5.data_processed[0]['strain'].values[-1], 99.6, 5)

        # test unsmoothed e modulus calculation
        tens_test_5.calc_e_modulus(smoothing=False)
        self.assertAlmostEqual(
            tens_test_5.results.loc[0, tens_test_5.e_modulus_title], 300, 5)

        # test plotting without any previous calculation, e.g. e modulus or
        # strength, including export
        tens_test_4.generate_plots(export_path='')

        # test errors
        self.assertRaises(
            ValueError, tensile_test_data.tensile_test, test_data, 'DataFrame',
            unit_strain='5')
        self.assertRaises(
            ValueError, tensile_test_data.tensile_test, 'test', 'import',
            column_names=['1', '2'], columns=[1])
        self.assertRaises(
            ValueError, tensile_test_data.tensile_test, 'test', 'impot',
            column_names=['1', '2'], columns=[1, 2])
        self.assertRaises(
            ValueError, tens_test_2.find_data_borders, onset_mode='2')
        self.assertRaises(
            ValueError, tens_test_2.find_data_borders, data_end_mode='percy')