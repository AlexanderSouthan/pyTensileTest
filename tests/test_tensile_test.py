#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import unittest

import numpy as np
import pandas as pd
from src.pyAnalytics import tensile_test_data


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
