# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:02:25 2023

@author: southan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyTensileTest import tensile_test
from little_helpers.math_functions import cum_dist_normal_with_rise


# import data from file
compr_test = tensile_test('compression test.xls', 'import', unit_strain='%',
                          offset_correction=True, start_sheet=0, columns=[0, 1, 2],
                          column_names=['tool_distance', 'strain', 'stress'], header_rows=1)


# Find measurement onset and sample destruction. The only function working for
# onset_mode='fit' is cum_dist_normal_with_rise because of the definition of
# the onset used in the code. This could be improved by allowing a more
# general definition.
compr_test.find_data_borders(
    onset_mode='fit', fit_function=cum_dist_normal_with_rise,
    fit_boundaries={'sigma': [0.01, 5], 'x_offset': [25, 35],
                    'slope': [0, 20], 'amp': [0, 500]},
    non_fit_par={'linear_rise': 'right'},
    onset_strain_range=[2, 40],
    data_end_mode='perc_drop', lower_thresh=0.05, drop_window=2)


compr_test.calc_e_modulus(r_squared_lower_limit=0.997)

# plot the raw data
fig1, ax1 = plt.subplots(2)
ax1[0].set_xlabel('strain [%]')
ax1[0].set_ylabel('stress [kPa]')
ax1[0].plot(compr_test.data_raw[0]['strain'], compr_test.data_raw[0]['stress'],
            label='raw data')
ax1[0].axvline(compr_test.onsets[0], c='g', ls='--',
               label='onset determined from fit on derivative')
ax1[0].axvline(compr_test.data_ends[0], c='r', ls='--',
               label='end of measurement determined from derivative threshold')
ax1[0].legend()
# plot the derivative
ax1[1].set_xlabel('strain [%]')
ax1[1].set_ylabel('stress derivative [kPa/%]')
ax1[1].set_ylim([-100, 1000])
ax1[1].plot(compr_test.data_revised[0]['strain'],
            compr_test.data_revised[0]['deriv_1'],
            label='derivative')
ax1[1].plot(compr_test.onset_fits[0]['x_fit'],
            compr_test.onset_fits[0]['y_fit'],
            label='fit of derivative')
ax1[1].axvline(compr_test.onsets[0], c='g', ls='--')
ax1[1].axvline(compr_test.data_ends[0], c='r', ls='--')
plt.tight_layout()

# plot the processed data
fig2, ax2 = plt.subplots()
ax2.set_xlabel('strain [%]')
ax2.set_ylabel('stress [kPa]')
ax2.plot(compr_test.data_processed[0]['strain'],
         compr_test.data_processed[0]['stress'],
         label='processed data')
# Plot line for linear fit for E modulus calculation
x_fit = [compr_test.data_processed[0]['strain'].iloc[0], compr_test.results['linear_limit [%]'].iloc[0]]
x_fit_range = np.diff(x_fit).item()
y_fit = [compr_test.data_processed[0]['stress'].iloc[0],
         compr_test.data_processed[0]['stress'].iloc[0]+x_fit_range*compr_test.results['slope_limit [kPa/%]'].iloc[0]]
ax2.plot(x_fit, y_fit, c='r', label='fit for E modulus with $R^2$ threshold')

ax2.legend()
