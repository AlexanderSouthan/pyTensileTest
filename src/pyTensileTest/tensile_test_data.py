# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:31:29 2020

@author: aso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from copy import deepcopy

from pyDataFitting.linear_regression import lin_reg_all_sections
from pyDataFitting.nonlinear_regression import (nonlinear_regression,
                                                calc_function)
from pyPreprocessing.smoothing import smoothing
from little_helpers.array_tools import closest_index
from little_helpers.num_derive import derivative


class tensile_test():
    def __init__(self, data, data_mode, unit_strain='%',
                 unit_stress='kPa', offset_correction=False, deriv_window=5,
                 **kwargs):
        """
        Initialize data container for tensile test data.

        Original raw data are stored in self.data_raw. The data is then
        streamlined and stored in self.data_revised.

        Parameters
        ----------
        data : string or DataFrame
            Either the DataFrame containing the data (data_mode is 'DataFrame')
            or otherwise the file path of the file that contains the tensile
            test data. The data will be imported to self.data_raw. The file
            will usually be an Excel file. The corresponding import filters
            have to be defined in self.import_data.
        data_mode : string
            Selects the import filter used when importing the data. Currently
            allowed values are 'import' and 'DataFrame'. 'DataFrame' means that
            the data is passed directly as data argument in a DataFrame with
            the format of self.data_raw. 'import' means the data is imported
            from an Excel file containing the data of different samples in
            different sheets. See kwargs section for import options.
        unit_strain : string, optional
            The unit of the strain values in the dataset. Allowed values are
            '', meaning that it is dimensionless, and '%', meaning that the
            strain is given in %. The default is '%'.
        unit_stress : string, optional
            The unit of the stress values in the dataset. The default is 'kPa'.
        offset_correction : bool, optional
            If True, the strain data is shifter by the offset correction
            method given by the corresponding kwarg. Default is False.
        deriv_window : int, optional
            The window used for averaging passed to the derivative function.
            Default is 5.
        **kwargs :
            offset_strain : list of float, optional
                A list containing two elements, the first smaller or equal to
                the second. They define the range of the strain data that is
                used to determine the offset that is subtracted from the stress
                values. The offset is calculated by averaging over all stress
                values at the strain values within the interval. If both values
                are equal, only one value is used for offset determination. The
                default is the first value of the measurement.
            start_sheet : int, optional
                Only needed for data_mode 'import'. The number of the first
                sheet in the Excel file containing the data, with 0 being the
                first sheet. This is useful if the first sheets contain e.g.
                some metadata about the measurements. All following sheets
                must contain one dataset each. The default is 2.
            columns : list of int, optional
                Only needed for data_mode 'import'. A list containing the
                number of the columns to be imported, with 0 being the first
                column. The default is [0, 1].
            column_names : list of string, optional
                Only needed for data_mode 'import'. Contains the names of the
                columns imported. Must at least contain 'stress' and 'strain'.
                'tool_distance' is also allowed if onset and end detection are
                performed (needed for recalculation of strain). Other names can
                be included, but are not used for anything at the moment. The
                default is ['strain', 'stress'].
            header_rows : int, optional
                Only needed for data_mode 'import'. The number of rows to be
                skipped during data import. The default is 2.

        Raises
        ------
        ValueError
            If no valid unit_strain or onset_mode is given.

        Returns
        -------
        None.

        """
        # strain units can be '%' or '' 
        self.unit_strain = unit_strain
        self.unit_stress = unit_stress
        self.offset_correction = offset_correction
        self.deriv_window = deriv_window
        self.onsets = []
        self.data_ends = []

        if self.offset_correction:
            self.offset_strain = kwargs.get('offset_strain', 'first')

        if self.unit_strain == '':
            self.strain_conversion_factor = 1
        elif self.unit_strain == '%':
            self.strain_conversion_factor = 100
        else:
            raise ValueError('No valid unit_strain. Allowed values are \'%\' '
                             'or \'\'.')

        self.e_modulus_title = 'e_modulus [' + self.unit_stress + ']'
        self.linear_limit_title = 'linear_limit [' + self.unit_strain + ']'
        self.strength_title = 'strength [' + self.unit_stress + ']'
        self.toughness_title = ('toughness [' + self.unit_stress + '] '
                                '(Pa = J/m^3)')
        self.elongation_at_break_title = ('elongation_at_break ['
                                          '' + self.unit_strain + ']')
        self.slope_limit_title = ('slope_limit [' + self.unit_stress + '/'
                                  '' + self.unit_strain + ']')
        self.intercept_limit_title = ('intercept_limit ['
                                      ''+ self.unit_stress + ']')

        self.results = pd.DataFrame([], columns=[
                'name', self.e_modulus_title, self.linear_limit_title,
                self.strength_title, self.toughness_title,
                self.elongation_at_break_title, self.slope_limit_title,
                self.intercept_limit_title])

        self.data_mode = data_mode

        if self.data_mode == 'import':
            start_sheet = kwargs.get('start_sheet', 2)
            columns = kwargs.get('columns', [0, 1])
            column_names = kwargs.get('column_names', ['strain', 'stress'])
            header_rows = kwargs.get('header_rows', 2)
            if len(column_names) != len(columns):
                raise ValueError('Number of imported columns and column names '
                                 'must be equal.')
            self.import_data(data, start_sheet, columns, column_names,
                             header_rows)
        elif self.data_mode == 'DataFrame':
            self.data_raw = data
        else:
            raise ValueError('No valid import mode entered. ALlowed values are'
                             '\'DataFrame\' and \'import\'.')
            
        # copy original raw data and use this copy for further processing
        self.data_revised = deepcopy(self.data_raw)
        for sample in self.data_revised:
            sample = self.streamline_data(sample)
            sample = self.append_derivative(sample, up_to_order=2)

        self.data_processed = deepcopy(self.data_revised)
        # for sample in self.data_processed:
        #     sample = self.find_data_borders(sample)

    def import_data(self, file_name, start_sheet=2, columns=[0, 1],
                    column_names=['strain', 'stress'], header_rows=2):
        """
        Import the data from the external sources into self.data_raw.

        Multiple datasets may be present in the source file given.
        self.data_raw is a list of DataFrames. Each dataframe contains the
        data of one tensile test. If only one dataset is present, it is a list
        containing only one DataFrame.

        For explanation of the arguments, see kwargs in docstring of
        self.__init__().

        Returns
        -------
        None.

        """
        # Read excel file
        raw_excel = pd.ExcelFile(file_name)
        # Save sheet data starting with start_sheet into list
        self.data_raw = []
        self.results['name'] = raw_excel.sheet_names[start_sheet:]
        for sheet_name in raw_excel.sheet_names[start_sheet:]:
            self.data_raw.append(
                    raw_excel.parse(sheet_name, header=header_rows,
                                    names=column_names,
                                    usecols=columns))

    def append_derivative(self, sample, up_to_order=2):
        for curr_order in range(1, up_to_order+1):
            sample['deriv_{}'.format(curr_order)] = derivative(
                sample['strain'].values/self.strain_conversion_factor,
                sample[['stress']].values.T, order=curr_order,
                averaging_window=self.deriv_window).T

        return sample

    def append_smoothed_data(self, sample, window=50):
        """
        Smoothes a dataset to allow a cleaner analysis of the slopes.

        Parameters
        ----------
        sample : DataFrame
            A dataset containing the stress-strain data of a single sample.
        window : int, optional
            The number of datapoints used around a specific datapoint for the
            rolling_median method. Default is 50.

        Returns
        -------
        sample : DataFrame
            The input DataFrame (sample) supplemented with a column containing
            the smoothed data.

        """
        smoothed_sample = sample.loc[:, 'stress'].values

        smoothed_sample = smoothing(
            smoothed_sample[np.newaxis], 'rolling_median', interpolate=False,
            point_mirror=True, window=window)

        sample['smoothed'] = np.squeeze(smoothed_sample.T)

        return sample

    def streamline_data(self, sample):
        """
        Streamline a dataset to make it fit for further analysis.
        
        Typical problems with real datasets are duplicate values, unsorted
        strain values (especially at the test start) and NaN values due to
        missing values in the source file. 

        Parameters
        ----------
        sample : DataFrame
            A dataset containing the stress-strain data of a single sample.

        Returns
        -------
        sample : DataFrame
            The streamlined dataset.

        """
        sample.drop_duplicates('strain', keep='first', inplace=True)
        sample.sort_values(by=['strain'], inplace=True)
        sample.dropna(inplace=True)
        if self.offset_correction:
            if self.offset_strain == 'first':
                sample['stress'] = sample['stress'] - sample['stress'].iloc[0]
            else:
                avg_mask = sample['strain'].between(*self.offset_strain)
                sample['stress'] = (sample['stress'] -
                                    sample['stress'].loc[avg_mask].mean())

        return sample

    def find_data_borders(self, onset_mode=None, data_end_mode=None, **kwargs):
        """
        Find the onset and end of measurement in test data.

        The onset is the strain at which the stress starts to increase.
        In some cases, it makes sense e.g. to start a compression test with
        some distance between the tool and the sample instead of using a
        preload to start data acquisition. In such cases, the onset needs to be
        calculated after the measurement and the strain values have to be
        recalculated. Recalculation can only be done if the tool distance
        during the test is also present in the dataset. Current methods are
        based on a fixed threshold in the stress values, on identifying the
        maximum value of the second derivative (maximum curvature), and
        perfoming a fit with a function.

        The end of the measurement is reached usually upon material failure.

        Parameters
        ----------
        onset_mode : string or None, optional
            Gives the method used to determine the onset of stress increase.
            Allowed values are 'stress_thresh' (onset at first datapoint above
            a given threshold), 'deriv_2_max' (onset at the maximum of the
            second derivative, i.e. place of maximum curvature), 'fit' (onset
            is determined from a regression) and None (no onset detection).
            The corresponding parameters are given by kwargs (see below).
            Default is None.
        data_end_mode : string or None, optional
            Gives the mthod to determine the end of the measurement, usually
            material failure. Allowed values are 'lower_thresh' (end of 
            measurement is identified by a large negative spike in the
            derivative), 'perc_drop' (end identified by a drop of stress
            relative to maximum stress in measurement) and None (no data end
            treatment). The corresponding methods are configured using kwargs
            (see below). Default is None.
        **kwargs :
            onset_strain_range : list of float, optional
                A list containing two floats defining the strain range where
                the onset is looked for. Default is the entire range of a test,
                so e.g. [0, 100] with unit_strain='%'.
            end_strain_range : list of float, optional
                A list containing two floats defining the strain range where
                the end of data is looked for. Default is the entire range of
                a compression test, so e.g. [0, 100] with unit_strain='%'.
            if onset_mode == 'stress_thresh':
                stress_thresh : float, optional
                    The stress theshold given in the unit of the stress used in
                    the dataset. All stress values smaller than stress_thresh
                    will be dropped from the dataset. The corresponding tool
                    distance will be defined as zero strain and the strain
                    values are recalculated accordingly. Thus, it only works
                    when a column exists within the dataset that contains the
                    tool distance during the tensile test. Default is 1.
            if onset_mode == 'fit':
                fit_function : string
                    The fit function used, must work with imported
                    nonlinear_regression function.
                fit_boundaries : list of tuples
                    The boundaries used for the fit. Must be a list of tuples,
                    each containing an upper and a lower limit for the fit
                    parameters. The order is given by the corresponding
                    imported function calc_function.
                fit_scale_factor : float, optional
                    A factor by which the data to be fitted is divided before
                    the fit. A higher value might result in a more stable/
                    reliable fitting process. Make sure to adapt the
                    fit_boundaries accordingly. Default is 1.
            if data_end_mode == 'lower_thresh' or 'perc_drop':
                lower_thresh : float, optional
                    the threshold level of the derivative at which the sample
                    is assumed to have failed. Default is -500 ('lower_thresh')
                    or 0.05 ('perc_drop').
            if data_end_mode == 'perc_drop':
                drop_window : int, optional
                    The number of datapoints used for a rolling sum of the
                    stress changes. A bigger window will allow to detect less
                    steep stress drops upon material failure. Default is 1,
                    meaning that the lower_thresh must be reached between two
                    individual data points.

        Returns
        -------
        None.

        """
        self.onset_mode = onset_mode
        self.data_end_mode = data_end_mode

        if self.onset_mode is not None:
            self.onset_strain_range = kwargs.get(
                'onset_strain_range', [0, self.strain_conversion_factor])
            if self.onset_mode == 'stress_thresh':
                self.stress_thresh = kwargs.get('stress_thresh', 1)
            elif self.onset_mode == 'deriv_2_max':
                pass
            elif self.onset_mode == 'fit':
                self.fit_function = kwargs.get('fit_function')
                self.fit_boundaries = kwargs.get('fit_boundaries')
                self.fit_scale_factor = kwargs.get('fit_scale_factor', 1)
                self.fit_params = []
                self.onset_fits = []
            else:
                raise ValueError('No valid onset_mode given.')

        if self.data_end_mode is not None:
            self.end_strain_range = kwargs.get(
                'end_strain_range', [0, self.strain_conversion_factor])
            if self.data_end_mode == 'lower_thresh':
                self.lower_thresh = kwargs.get('lower_thresh', -500)
            elif self.data_end_mode == 'perc_drop':
                self.lower_thresh = kwargs.get('lower_thresh', 0.05)
                self.drop_window = kwargs.get('drop_window', 1)
            else:
                raise ValueError('No valid data_end_mode given.')

        for sample in self.data_processed:
            assert 'tool_distance' in sample.columns, (
                'There is no column with a label \'tool_distance\' in the '
                'dataset. Please make sure that this is the case before '
                'calling this method.')

            # onset identification
            if self.onset_mode is not None:
                # select only the data between the corresponding onset strain
                # borders
                data_mask = sample['strain'].between(*self.onset_strain_range)

                if self.onset_mode == 'stress_thresh':
                    data_mask *= sample['stress'] >= self.stress_thresh
                    onset_idx = data_mask.idxmax()

                elif self.onset_mode == 'deriv_2_max':
                    onset_idx = sample.loc[data_mask, 'deriv_2'].idxmax()

                elif self.onset_mode == 'fit':
                    x_for_fit = sample.loc[data_mask,'strain'].values
                    y_for_fit = (sample.loc[data_mask,'deriv_1'].values/
                                 self.fit_scale_factor)
                    self.fit_params.append(nonlinear_regression(
                        x_for_fit, y_for_fit, self.fit_function,
                        boundaries=self.fit_boundaries, max_iter=1000).x)  # sigma, x_offset, slope, amp
                    self.onset_fits.append(pd.DataFrame(np.array([
                        x_for_fit, calc_function(
                            x_for_fit, self.fit_params[-1], self.fit_function
                            )*self.fit_scale_factor]).T,
                        columns=['x_fit', 'y_fit']))

                    # onset is defined as x_offset + sigma. This only makes sense
                    # when self.fit_function is 'cum_dist_normal_with_rise'. For
                    # other cases, the next calculation must be adapted.
                    curr_onset = self.fit_params[-1][1] + self.fit_params[-1][0]

                    onset_idx = sample['strain'].index[closest_index(
                        curr_onset, sample['strain'].values)].values[0]
            else:
                onset_idx = sample.index[0]

            # data end/sample failure identification
            if self.data_end_mode is not None:
                # prevent that the found end of data is at a smaller strain
                # than the onset
                if sample.at[onset_idx, 'strain'] > self.end_strain_range[0]:
                    self.end_strain_range[0] = sample.at[onset_idx, 'strain']

                data_mask = sample['strain'].between(*self.end_strain_range)
                if self.data_end_mode == 'lower_thresh':
                    data_mask *= sample['deriv_1'] < self.lower_thresh
                    if any(data_mask):
                        end_idx = data_mask.idxmax()
                    else:
                        warnings.warn('No end of data found. Possibly the '
                                      'threshold used is not good. Using the '
                                      'last data point instead.')
                        end_idx = sample.index[-1]
                elif self.data_end_mode == 'perc_drop':
                    diffs = pd.Series(
                        np.diff(sample['stress']), index=sample.index[:-1]
                        ).rolling(self.drop_window).sum()
                    perc_drop = diffs / sample['stress'].max()

                    thresh_violated = data_mask[:-1]*(
                        (np.abs(perc_drop) > self.lower_thresh) &
                        (perc_drop < 0))

                    if any(thresh_violated):
                        end_idx = thresh_violated.idxmax()
                    else:
                        warnings.warn('No end of data found. Possibly the '
                                      'threshold used is not good. Using the '
                                      'last data point instead.')
                        end_idx = sample.index[-1]
            else:
                end_idx = sample.index[-1]

            # store identified data borders in corresponding lists
            self.onsets.append(sample.at[onset_idx, 'strain'])
            self.data_ends.append(sample.at[end_idx, 'strain'])

            # get the tool distance at the onset
            h_0 = sample.at[onset_idx, 'tool_distance']

            # crop dataset according to the identified data borders
            onset_loc = sample.index.get_loc(onset_idx)
            sample.drop(sample.index[:onset_loc], inplace=True)
            end_loc = sample.index.get_loc(end_idx)
            sample.drop(sample.index[end_loc:], inplace=True)

            # recalculate the strain and the derivatives
            sample['strain'] = ((h_0-sample['tool_distance'])/h_0 *
                                self.strain_conversion_factor)
            sample = self.append_derivative(sample, up_to_order=2)

    def calc_e_modulus(self, r_squared_lower_limit=0.995, lower_strain_limit=0,
                       upper_strain_limit=50, smoothing=True, **kwargs):
        """
        Calculate the E modulus from the tensile test data.

        The fitting algorithm works in such a way that it starts a linear
        regression with the first two datapoints from which the slope and the
        coefficient of determination (R^2) is obtained. This procedure is
        repeated, adding one datapoint for each iteration, until
        upper_strain_limit is reached. The E modulus is then determined by the
        point where R^2 is still greater than r_squared_lower_limit.

        Parameters
        ----------
        r_squared_lower_limit : float, optional
            The R^2 limit. Defines the value of R^2 that is still acceptable
            for the fit to be considered linear. The default is 0.995.
        lower_strain_limit : float, optional
            The lower strain limit used for the fit. The default is 0.
        upper_strain_limit : float, optional
            The upper strain limit used for the fit. It might make sense to
            give a reasonably low limit if computation speed is important. The
            number of linear fit that have to be performed for each dataset is
            equal to the number of datapoints between lower_strain_limit and
            upper_strain_limit. The default is 50.
        smoothing : boolean, optional
            Defines if the dataset is smoothed before the calculations. Usually
            this is necessary. Smoothing is controlled by kwargs explained
            below, a Savitzky Golay method is used. The default is True. 
        **kwargs : 
            window : int, optional
                The number of datapoints used around a specific datapoint for
                the rolling_median method. Default is 50.

        Returns
        -------
        DataFrame
            The current results DataFrame.

        """
        e_modulus = []
        slope_limit = []
        intercept_limit = []
        linear_limit = []
        self.r_squared_lower_limit = r_squared_lower_limit

        # Data pre-processing for E modulus calculation
        for sample in self.data_processed:
            if smoothing:
                window = kwargs.get('window', 50)
                sample = self.append_smoothed_data(sample, window)
            else:
                # This is a little misleading because in this case, the data is
                # not smoothed. However, the column is needed for further
                # analysis.
                sample['smoothed'] = sample['stress']

        self.data_processed_cropped = []
        # max_strain_cropped = 0
        for sample in self.data_processed:
            sample_cropped = sample.copy()
            # Extract relevant strain range for youngs modulus
            indexNames = sample_cropped[(
                sample_cropped['strain'] <= lower_strain_limit)].index
            indexNames_2 = sample_cropped[(
                sample_cropped['strain'] >= upper_strain_limit)].index
            sample_cropped.drop(indexNames, inplace=True)
            sample_cropped.drop(indexNames_2, inplace=True)
            sample_cropped.reset_index(drop=True, inplace=True)

            # do regression and append results to sample DataFrame
            regression_results = lin_reg_all_sections(
                sample_cropped.loc[:, 'strain'].values,
                sample_cropped.loc[:, 'smoothed'].values,
                r_squared_limit=r_squared_lower_limit, mode='both')
            sample_cropped = pd.concat([sample_cropped, regression_results[0]],
                                       axis=1)

            slope_limit.append(regression_results[3])
            intercept_limit.append(regression_results[4])
            linear_limit.append(regression_results[1])

            # Calculate youngs modulus
            curr_e_modulus = (np.around(regression_results[3], decimals=3)*
                              self.strain_conversion_factor)

            # Save result in list
            e_modulus.append(curr_e_modulus.round(3))
            self.data_processed_cropped.append(sample_cropped)

        self.results[self.e_modulus_title] = e_modulus
        self.results[self.linear_limit_title] = linear_limit
        self.results[self.slope_limit_title] = slope_limit
        self.results[self.intercept_limit_title] = intercept_limit
        # return self.results

    def calc_strength(self):
        """
        Calculate the sample strengths.

        Calculation is done simply by selecting the maximum stress observed in
        the datasets.

        Returns
        -------
        list
            A list containing the calculation results.

        """
        self.strength = []
        for sample in self.data_processed:
            self.strength.append(
                    np.around(sample['stress'].max(), decimals=1))

        self.results[self.strength_title] = self.strength
        return self.strength

    def calc_toughness(self):
        """
        Calculate the sample toughnesses.

        Calculation is done by integrating over the stress-strain curves.

        Returns
        -------
        list
            A list containing the calculation results.

        """
        self.toughness = []
        for sample in self.data_processed:
            self.toughness.append(
                    np.around(np.trapz(sample['stress'],
                                       x=sample['strain'])/
                self.strain_conversion_factor, decimals=1))
        self.results[self.toughness_title] = self.toughness
        return self.toughness

    def calc_elongation_at_break(self):
        """
        Calculate the elongation at break of the samples.

        Calculation is done simply by selecting the maximum strain observed in
        the datasets.

        Returns
        -------
        list
            A list containing the calculation results.

        """
        self.elongation_at_break = []
        for sample in self.data_processed:
            self.elongation_at_break.append(np.around(
                sample['strain'].at[sample['stress'].idxmax()], decimals=1))
        self.results[self.elongation_at_break_title] = self.elongation_at_break
        return self.elongation_at_break

    def generate_plots(self, export_path=None, **kwargs):
        if (self.results[self.e_modulus_title].isna().values.any()
        ) or (self.results[self.linear_limit_title].isna().values.any()):
            r_squared_lower_limit = kwargs.get('r_squared_lower_limit', 0.995)
            lower_strain_limit = kwargs.get('lower_strain_limit', 0)
            upper_strain_limit=kwargs.get('upper_strain_limit', 15)
            smoothing = kwargs.get('smoothing', True)
            sav_gol_window = kwargs.get('sav_gol_window', 500)
            sav_gol_polyorder = kwargs.get('sav_gol_polyorder', 2)
            data_points=kwargs.get('data_points', 10000)

            self.calc_e_modulus(r_squared_lower_limit=r_squared_lower_limit,
                                lower_strain_limit=lower_strain_limit,
                                upper_strain_limit=upper_strain_limit,
                                smoothing=smoothing,
                                sav_gol_window=sav_gol_window,
                                sav_gol_polyorder=sav_gol_polyorder,
                                data_points=data_points)

        if self.results[self.strength_title].isna().values.any():
            self.calc_strength()
        if self.results[self.toughness_title].isna().values.any():
            self.calc_toughness()
        if self.results[self.elongation_at_break_title].isna().values.any():
            self.calc_elongation_at_break()

        global_min_stress = 0
        global_max_stress = 0
        global_min_strain = 0
        global_max_strain = 0

        for sample in self.data_processed:
            if sample['stress'].max() > global_max_stress:
                global_max_stress = sample['stress'].max()
            if sample['strain'].max() > global_max_strain:
                global_max_strain = sample['strain'].max()
            if sample['stress'].min() < global_min_stress:
                global_min_stress = sample['stress'].min()
            if sample['strain'].min() < global_min_strain:
                global_min_strain = sample['strain'].min()

        max_strain_cropped = 0
        for sample in self.data_processed_cropped:
            if sample['strain'].iloc[-1] > max_strain_cropped:
                max_strain_cropped = sample['strain'].iloc[-1]

        # Plot whole graph with regression line
        for ii, (raw_sample, processed_sample, el_at_break, strength,
                curr_linear_limit, curr_slope_limit, curr_intercept_limit
                ) in enumerate(zip(
                    self.data_processed, self.data_processed,
                    self.elongation_at_break, self.strength,
                    self.results[self.linear_limit_title],
                    self.results[self.slope_limit_title],
                    self.results[self.intercept_limit_title])):
            fig = plt.subplot(len(self.data_processed), 3, 1+ii*3)
            if ii==0: plt.title('Tensile test')
            if ii==int(len(self.data_processed)/2):
                plt.ylabel(r'$\sigma$ [kPa]')
            if ii==len(self.data_processed)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            plt.axvline(el_at_break, ls='--', c='k', lw=0.5)
            plt.axhline(strength, ls='--', c='k', lw=0.5)
            plt.plot(raw_sample['strain'], raw_sample['stress'], linestyle='-')
            plt.plot(processed_sample['strain'], processed_sample['stress'],
                     linestyle='-', color='y')
            plt.plot(np.linspace(0, curr_linear_limit),
                     curr_slope_limit*np.linspace(0, curr_linear_limit)+
                     curr_intercept_limit, color='indianred')
            plt.xlim(global_min_strain, 1.05*global_max_strain)
            plt.ylim(global_min_stress, 1.05*global_max_stress)

        # Plot r-sqaured 
        plt.subplots_adjust(wspace = 0.5)
        for ii,(emod_sample,curr_linear_limit) in enumerate(
                zip(self.data_processed_cropped,
                    self.results[self.linear_limit_title])):
            fig = plt.subplot(len(self.data_processed), 3, 2+ii*3)
            if ii==0: plt.title('Coefficient of determination')
            if ii==int(len(self.data_processed)/2): plt.ylabel('$R^2$')
            if ii==len(self.data_processed)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            plt.yticks([0.975, 1.0])
            plt.plot(emod_sample.loc[1:, 'strain'],
                     emod_sample.loc[1:, 'r_squared'], color='indianred')
            plt.axvline(curr_linear_limit, ls='--', c='k', lw=0.5)
            plt.axhline(self.r_squared_lower_limit, ls='--', c='k', lw=0.5)
            plt.xlim(0, max_strain_cropped)
            plt.ylim(0.95, 1)

        # Plot E modulus
        for ii,(emod_sample,curr_linear_limit) in enumerate(
                zip(self.data_processed_cropped,
                    self.results[self.linear_limit_title])):
            fig = plt.subplot(len(self.data_processed), 3, 3+ii*3)
            if ii==0: plt.title('E modulus')
            if ii==int(len(self.data_processed)/2): plt.ylabel('$E$ [kPa]')
            if ii==len(self.data_processed)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            #plt.ylim(0.95, 1)
            #plt.yticks([0.975,1.0])
            plt.plot(emod_sample.loc[1:,'strain'],
                     emod_sample.loc[1:,'slopes']*100, color='indianred')
            plt.axvline(curr_linear_limit,ls='--', c='k', lw=0.5)
            plt.xlim(0, max_strain_cropped)

        if export_path is not None:
            export_name = kwargs.get('export_name', 'Tensile test')
            # Save figure as svg
            if not os.path.exists(export_path + 'Plots/'):
                os.makedirs(export_path + 'Plots/')
            plt.savefig(export_path + 'Plots/' + export_name + '.png', dpi=500)
            #plt.clf()
            plt.close()
