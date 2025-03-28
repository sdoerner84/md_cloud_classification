'''
Created on Fri Dec 11 16:47:06 2020

@authors:
Vinod Kumar
Lucas Reischmann
Steffen Beirle
Steffen Ziegler

Required modules (tested with version):
Python 3.11.11
numpy 2.2.2
scipy 1.15.1
matplotlib 3.10.0

Assumption: Having the data as array of the form SCANS x ELEVATIONS
'''
import warnings
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from md_cloud_classification.toolbox import MDCCError, time_conversions as tc
from md_cloud_classification.toolbox.md_cloud_result import MDCloudResult


__version__ = '1.0.3'


def gauss(x: float, offset: float, scale: float, mu: float, sigma: float):
    '''
    Gauss function used for fitting in the MAX-DOAS cloud classification
    calibration routine (color index CI and oxygen dimer O4).
    '''
    return offset + scale * np.exp(-((x - mu)**2 / (2. * sigma**2)))


def get_daily_idc_before_after(dt: np.array, elev: np.array,
                               inst_lon: np.array, select_elev: float):
    '''
    Assuming predefined sequences in all cases.

    Data variables:
    @dt datetime of a measurement
    @elev measured elevation angle
    @inst_lon longitude of the instrument location, needed for calculating the
        solar day and finding daily indices
    All data variables are required to have the same shape/size
    @select_elev elevation angle for which the TSI should be derived

    RETURNS Indices of the selected elevation at time step t-1 (before) and
        t+1 (after). The elevation at time step t is given in the elev
        parameter
    '''
    vec_solar_dt = np.vectorize(tc.calc_solar_dt)
    vec_dt_on_day = np.vectorize(tc.check_dt_on_day)
    solar_dt = vec_solar_dt(dt, inst_lon)
    date_arr = [dt.replace(hour=0, minute=0, second=0, microsecond=0)
                for dt in solar_dt.compressed()]
    solar_dates = np.unique(date_arr)
    result_idc = {}
    for solar_date in solar_dates:
        datestr = solar_date.strftime('%Y-%m-%d')
        result_idc[datestr] = {}
        scan_idc, elev_idc = np.where((elev == select_elev) &
                                      vec_dt_on_day(dt, solar_date))
        if len(scan_idc) == 0:
            result_idc[datestr]['cur_idc'] = None
            result_idc[datestr]['prev_idc'] = None
            result_idc[datestr]['next_idc'] = None
            continue
        # Special treatment for first and last entry
        scan_idc = np.insert(scan_idc, [0, -1], [scan_idc[0], scan_idc[-1]])
        elev_idc = np.insert(elev_idc, [0, -1], [elev_idc[0], elev_idc[-1]])
        # Define current, previous and next elevation indices
        # Default Case (FRM4DOAS):
        # one angle per sequence
        # all entries in a sequence in chronological order
        result_idc[datestr]['cur_idc'] = (scan_idc[1:-1], elev_idc[1:-1])
        result_idc[datestr]['prev_idc'] = (scan_idc[:-2], elev_idc[:-2])
        result_idc[datestr]['next_idc'] = (scan_idc[2:], elev_idc[2:])
    return result_idc


def get_idc(elev, select_elev, order='first'):
    '''
    Get indices of a chosen elevation angle in a given array of elevation
    angles. If an elevation occurs multiple times within a scan, take the
    first/last/all occurrences.

    @elev measured elevation angle
    @select_elev elevation angle for which the indices should be searched
    @order (default=first) describes which zenith measurement is selected:
        first - returning first entry of select_elev for each scan
        last - returning last entry of select_elev for each scan
        all - returning all occurrences of selected_elev for each scan

    RETURNS a tuple of indices matching the selected elevation
    '''
    elev_mask = elev == select_elev
    if order == 'first':
        elev_idc = np.argmax(elev_mask, axis=1)
    elif order == 'last':
        elev_idc = elev.shape[1] - 1 - np.argmax(elev_mask[:, ::-1], axis=1)
    elif order == 'all':
        return np.where(elev_mask)
    else:
        errmsg = f"Method (order={order}) not defined. "
        errmsg += "Select 'first' or 'last'"
        raise MDCCError(errmsg)
    # Correct for cases where select_elev is not in a scan.
    elev_idc[~elev_mask.any(axis=1)] = -1
    # Get scan idc for matching scans
    scan_idc = np.arange(elev.shape[0])
    scan_idc = scan_idc[elev_idc != -1]
    elev_idc = elev_idc[elev_idc != -1]
    return (scan_idc, elev_idc)


def fill_masked_array(any_array, fillvalue=False) -> np.ndarray:
    '''
    Make sure any array is a np.ndarray. If it is an instance of masked array,
    use the given fill value to convert the masked array to a np.ndarray

    @any_array
    @fillvalue (optional)
    '''
    if isinstance(any_array, np.ma.MaskedArray):
        return any_array.filled(fillvalue)
    return any_array


class MAXDOASCloudClassification():
    '''
    Cloud classification algorithm following the methods described in
    Wagner et al., 2014 (https://doi.org/10.5194/amt-7-1289-2014)
    Wagner et al., 2016 (https://doi.org/10.5194/amt-9-4803-2016)
    Wagner et al., 2024 FRM4DOAS D2.2 ATBD
    '''
    def __init__(self, cc_config: dict, thresholds_config: dict):
        '''
        @cc_config is a dictionary of settings used to configure the cloud
            classification. All necessary parameters for the configuration file
            can be found in the example_config.yml
        @thresholds is a dictionary of threshold settings. All necessary
            parameters can be found in the default_thresholds.yml
        '''
        self.config = cc_config
        if 'normalization_ci' not in cc_config:
            self.config['normalization_ci'] = None
        if 'normalization_o4' not in cc_config:
            self.config['normalization_o4'] = None
        required_configs = ['zenith_elevation', 'delta_time_scan',
                            'classification_elev_range',
                            'classification_sza_range',
                            'valid_spread_sza_range',
                            'normalize_ci_sza_range', 'normalize_ci_range',
                            'normalize_ci_bin_size',
                            'normalize_o4_sza_range', 'normalize_o4_range',
                            'normalize_o4_bin_size']
        missing_configs = np.setdiff1d(required_configs,
                                       list(cc_config.keys()))
        if len(missing_configs) > 0:
            msg = "Missing configuration entries for cloud classification: "
            msg += ",".join(missing_configs)
            self.print_log(msg, raise_exception=MDCCError)
        if 'ignore_elev' not in self.config:
            self.config['ignore_elev'] = []
        elif self.config['ignore_elev'] is None:
            self.config['ignore_elev'] = []
        self.threshold_config = thresholds_config
        self.thresholds = None
        self.classification_mask = None

    def print_log(self, msg: str, raise_exception: Exception=None):
        '''
        Print information to console/file with current time stamp information.

        @msg
        @raise_exception (default=None) is set to any Exception class if the
            given message should raise an exception.
        '''
        msg = datetime.now().strftime('%Y-%m-%d %H:%M:%S >>> ') + msg
        print(msg)
        if raise_exception is not None:
            raise raise_exception(msg)

    def check_ci_normalization(self):
        '''
        Check if CI normalization factor is valid. Raise error, if not.
        '''
        if self.config['normalization_ci'] is None:
            errmsg = 'CI normalization value is None. Provide '
            errmsg += 'normalization_ci in the config '
            errmsg += 'or run the normalize_ci(...) function.'
            self.print_log(errmsg, raise_exception=MDCCError)

    def check_o4_normalization(self):
        '''
        Check if O4 normalization factor is valid. Raise error, if not.
        '''
        if self.config['normalization_o4'] is None:
            errmsg = 'O4 normalization value is None. Provide '
            errmsg += 'normalization_o4 in the config '
            errmsg += 'or run the normalize_o4(...) function.'
            self.print_log(errmsg, raise_exception=MDCCError)

    def check_classification_mask(self):
        '''
        Check if classification mask is set. Raise error, if not.
        '''
        if self.classification_mask is None:
            errmsg = 'Classification_mask is not set. Run '
            errmsg += 'set_classification_mask(...)'
            self.print_log(errmsg, raise_exception=MDCCError)

    def check_thresholds(self):
        '''
        Check if thresholds are set. Raise error, if not.
        '''
        if self.thresholds is None:
            errmsg = "Run the gen_thresholds(...) routine before running this "
            errmsg += "function."
            self.print_log(errmsg, raise_exception=MDCCError)

    def check_shape(self, *args):
        '''
        Checks if the shape of all given arguments agree. Raises an error if
        not.
        '''
        if len(args) < 2:
            return
        first_shape = args[0].shape
        for arg in args[1:]:
            if arg.shape != first_shape:
                errmsg = "Not all given data array have the same shape. Data "
                errmsg += "array contain at least two different shapes: "
                errmsg += f"{first_shape} and {arg.shape}."
                self.print_log(errmsg, raise_exception=MDCCError)

    def set_classification_mask(self, sza: np.array, elev: np.array):
        '''
        Set a mask that defines for which measurements contribute to the cloud
        classification algorithm. The mask is set as class variable.

        Data variables:
        @sza solar zenith angle
        @elev measured elevation angle
        All data variables are required to have the same shape/size

        RETURNS
        '''
        self.check_shape(sza, elev)
        sza_mask = sza < self.config['classification_sza_range'][0]
        sza_mask |= sza > self.config['classification_sza_range'][1]
        elev_mask = elev < self.config['classification_elev_range'][0]
        elev_mask |= elev > self.config['classification_elev_range'][1]
        for ignore_elev in self.config['ignore_elev']:
            elev_mask |= (elev == ignore_elev)
        self.classification_mask = sza_mask | elev_mask
        return self.classification_mask

    def gen_thresholds(self, sza: np.array, elev: np.array):
        '''
        Generating current thresholds based on the given universal polynomials
        (see Wagner et al. 2014)
        Update - Wagner et al., 2016
        Update - Constantly cloudy/clear categories, 2024
        See threshold configuration file for detailed descriptions of the
        threshold values.

        Data variables:
        @sza solar zenith angle
        @elev measured elevation angle
        All data variables are required to have the same shape/size

        Generated thresholds are stored in the class variable "thresholds"
        and returned as a dictionary.
        '''
        if self.thresholds is not None:
            wrnmsg = "Thresholds are already defined. Returning pre-defined "
            wrnmsg += "thresholds."
            self.print_log(wrnmsg)
            return self.thresholds
        self.check_shape(sza, elev)
        zenith_idc = get_idc(elev, self.config['zenith_elevation'],
                             order='last')
        th_cfg = self.threshold_config
        sza_zenith = sza[zenith_idc]
        # Filter all values outside of defined SZA range
        sza_filter = sza_zenith < self.config['classification_sza_range'][0]
        sza_filter |= sza_zenith > self.config['classification_sza_range'][1]
        sza_zenith[sza_filter] = np.nan
        # Calculation of SZA dependent or constant thresholds
        # for a given data set
        th = {}
        # Polynomial depending on SZA / zenith_elevation:
        th_keys = ['CI_TH',  # CI Threshold value
                   'CI_MIN',  # CI Minimum value
                   'O4_TH']  # O4 AMF zenith +0.85 for threshold
        sza_norm = sza_zenith / self.config['zenith_elevation']
        for th_key in th_keys:
            th[th_key] = np.poly1d(th_cfg[th_key])(sza_norm)
        # Polynomial depending on SZA only
        th_keys = ['CI_AOD02',  # CI_AOD02 - Normalization for TSI calculation
                   'AVG_SZA']  # AVG_SZA - Average SZA dependence of spread for AOD 0.1 and 0.2
        for th_key in th_keys:
            th[th_key] = np.poly1d(th_cfg[th_key])(sza_zenith)
        # Spread is only valid in this SZA range
        spread_sza_filter = sza_zenith < self.config['valid_spread_sza_range'][0]
        spread_sza_filter |= sza_zenith > self.config['valid_spread_sza_range'][1]
        th['AVG_SZA'][spread_sza_filter] = np.nan
        # Constant THs:
        th_keys = ['SPREAD_CI',  # Spread CI threshold
                   'SPREAD_O4',  # Spread O4 threshold
                   'O4_TH_OFFSET'
                   ]
        for th_key in th_keys:
            th[th_key] = np.zeros_like(sza_zenith) + th_cfg[th_key]
        # TSI_TH - CI Temporal smoothness indicator (TSI)
        th['TSI_TH'] = np.poly1d(th_cfg['TSI_TH'])(sza_norm)
        th['TSI_TH'] /= th['CI_AOD02']
        th['TSI_TH'] *= th_cfg['TSI_TH_ALPHA'] * th_cfg['TSI_CONST_FACTOR']
        # TSI_CONST_TH - CI Temporal smoothness indicator for constantly
        # cloudy and constantly clear conditions (TSI)
        th['TSI_CONST_TH'] = np.poly1d(th_cfg['TSI_TH'])(sza_norm)
        th['TSI_CONST_TH'] /= th['CI_AOD02']
        th['TSI_CONST_TH'] *= th_cfg['TSI_TH_ALPHA']
        self.thresholds = th
        return th

    def normalize_ci(self, sza: np.array, elev: np.array, ci: np.array,
                     plot_stream: PdfPages=None, method: str='peak_finding',
                     verbose: bool=False):
        '''
        Derive a normalization factor for the color index in order to make
        the given threshold values applicable to different instrument.

        Requires gen_thresholds(...) to be run before.

        Data variables:
        @sza solar zenith angle
        @elev measured elevation angle
        @ci color index of radiance at 330nm / radiance at 390nm
        All data variables are required to have the same shape/size
        @plot_stream (default=None) if given, the normalization plots will be
            created on the given PdfPages file stream
        @method different methods to derive the CI normalization parameter are
            implemented:
            'gauss_fit' - described and used in Wagner et al., 2016
            'peak_finding' - implemented by Vinod Kumar and used in FRM4DOAS
                validation (DEFAULT)
        @verbose (default=False) if set True, normalization output will be
            printed

        RETURNS color index normalization value (divide given CI by this value)
        '''
        if self.config['normalization_ci'] is not None:
            if verbose:
                msg = "Normalization will only be performed if "
                msg += "'normalization_ci' is not set in the configuration "
                msg += f"file. Using: {self.config['normalization_ci']} "
                msg += "from configuration file."
                self.print_log(msg)
            return self.config['normalization_ci']
        self.check_thresholds()
        self.check_shape(sza, elev, ci)
        # Select indices of zenith elevation measurements
        zenith_idc = get_idc(elev, self.config['zenith_elevation'], order='last')
        ci_norm = ci[zenith_idc] / self.thresholds['CI_MIN']
        norm_bin_range = tuple(self.config['normalize_ci_range'])
        norm_bin_size = self.config['normalize_ci_bin_size']
        norm_bins = np.arange(*norm_bin_range, norm_bin_size)
        center_bins = 0.5 * (norm_bins[:-1] + norm_bins[1:])
        # First gauss fit to clip filtered values
        sza_filter = sza[zenith_idc] > self.config['normalize_ci_sza_range'][0]
        sza_filter &= sza[zenith_idc] < self.config['normalize_ci_sza_range'][1]
        count_abs, _ = np.histogram(ci_norm[sza_filter], norm_bins)
        count_rel = count_abs / np.sum(count_abs)
        if plot_stream is not None:
            fig, ax = plt.subplots(figsize=[8, 4])
            ax.plot(center_bins, count_rel, c='k', alpha=0.7,
                    label='CI distribution')
        # Check if CI values can be fitted with a normal distribution
        try:
            # Start values for fit
            offset = 0
            scale = np.nanmax(count_rel)
            mu = center_bins[np.argmax(count_rel)]
            sigma = 0.2
            fit_params, _ = curve_fit(gauss, center_bins, count_rel,
                                      [offset, scale, mu, sigma])
        except RuntimeError:
            msg = 'Could not fit a gauss curve to the given frequency '
            msg += 'distribution of the color index. Please check radiance '
            msg += 'values.'
            if plot_stream is not None:
                plot_stream.savefig(fig, dpi=200, bbox_inches='tight')
                plt.close(fig)
            self.print_log(msg, raise_exception=ValueError)
        ###
        if method == 'gauss_fit':
            # Method description:
            # Final gauss fit after clipping using fitted mu and sigma from
            # test fit to remove all values above mu + 1 * sigma
            ci_filter = ci_norm < fit_params[2] + fit_params[3]
            ci_selected = ci_norm[(ci_filter) & (sza_filter)]
            count_abs, _ = np.histogram(ci_selected, norm_bins)
            count_rel = count_abs / np.sum(count_abs)
            fit_params, _ = curve_fit(gauss, center_bins, count_rel,
                                      [offset, scale, mu, sigma])
            normalization_ci = fit_params[2]
            if plot_stream is not None:
                # ADD fitted calibration CI to plot.
                fit_line = gauss(center_bins, *tuple(fit_params))
                ax.plot(center_bins, fit_line, '-', color='r',
                        label='gauss fit')
        elif method == 'peak_finding':
            # Method description:
            #   1) Use find_peaks to find all local maxima in the color index
            #      frequency distribution.
            #   2) Remove all peaks with a local maximum value below 10% of the
            #      global maximum value.
            #   3) From the remaining peaks, select the one with in the lowest
            #      center_bin.
            # Using count_rel from binning after only filtering for SZA
            peaks, _ = find_peaks(count_rel, prominence=0.05*np.max(count_rel))
            peak_filter = count_rel[peaks] > 0.1 * np.max(count_rel[peaks])
            normalization_ci = center_bins[np.min(peaks[peak_filter])]
            if plot_stream is not None:
                # ADD peaks and selected calibration CI to plot
                ax.scatter(center_bins[peaks], count_rel[peaks], marker='o',
                           s=24, facecolor="none", edgecolor='b',
                           label='PF: All peaks')
                ax.scatter(center_bins[peaks[peak_filter]],
                           count_rel[peaks[peak_filter]],
                           marker='x', c='r', s=24, label='PF: $>$ 0.1 * max')
                ci_norm_label = 'PF: CI norm. value\n'
                ci_norm_label += f'CINorm={normalization_ci:.2f}'
                ax.axvline(center_bins[np.min(peaks[peak_filter])], color='r',
                           label=ci_norm_label)
        frac_below = np.sum(count_rel[center_bins < normalization_ci])
        # INTENDED CHANGE: Use CI instead of 1 / CI as the found normalization
        # CI - it is much more intuitive to see the same value as in the plots
        # BUT REMEMBER TO divide by this value, instead of multiplying it.
        self.config['normalization_ci'] = normalization_ci
        if verbose:
            calib_msg = 'Color index (CI) normalization value is '
            calib_msg += f'{normalization_ci:.3f}\nFraction of CI below the '
            calib_msg += f'threshold is {frac_below * 100:.2f}%.'
            self.print_log(calib_msg)
        if plot_stream is not None:
            ax.legend(loc='upper right')
            ax.set_xlabel('Normalized CI')
            ax.set_ylabel('Relative frequency')
            if 'site' in self.config:
                ax.annotate(self.config['site'], xy=(0.02, 0.9),
                            xycoords='axes fraction')
            ci_fraction_note = 'Fraction of CI\nbelow '
            ci_fraction_note += f'threshold: {frac_below * 100:.2f}%'
            ax.annotate(ci_fraction_note, xy=(0.02, 0.8),
                        xycoords='axes fraction')
            ax.grid(alpha=0.4)
            plt.tight_layout()
            plot_stream.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close()
        return self.config['normalization_ci']

    def normalize_o4(self, sza: np.array, elev: np.array, o4_damf: np.array,
                     cloud_type: MDCloudResult, plot_stream: PdfPages=None,
                     verbose: bool=False):
        '''
        Function to retrieve a normalization factor for the O4 differential
        air mass factor (DAMF).

        Requires gen_thresholds(..) to be run before.

        Data variables:
        @sza solar zenith angle
        @elev measured elevation angle
        @o4_damf measured elevation angle
        All data variables are required to have the same shape/size

        @cloud_type MDCloudResult object (see toolbox/md_cloud_result.py)
        @verbose (default=False) if set True, normalization output will be
            printed

        RETURNS O4 DAMF normalization value
        '''
        if self.config['normalization_o4'] is not None:
            if verbose:
                msg = "Normalization will only be performed if "
                msg += "'normalization_o4' is not set in the configuration "
                msg += f"file. Using: {self.config['normalization_o4']} from "
                msg += "configuration file."
                self.print_log(msg)
            return self.config['normalization_o4']
        self.check_thresholds()
        self.check_shape(sza, elev, o4_damf)
        # Perform O4 calibration only if CI classification did not raise flags
        # for broken clouds or cloud holes
        cloud_mask = cloud_type.main[:, 3] == 0
        cloud_mask &= cloud_type.main[:, 4] == 0
        # Only for a given SZA range
        zenith_idc = get_idc(elev, self.config['zenith_elevation'],
                             order='last')
        sza_mask = sza[zenith_idc] > self.config['normalize_o4_sza_range'][0]
        sza_mask &= sza[zenith_idc] < self.config['normalize_o4_sza_range'][1]
        finite_mask = np.isfinite(o4_damf[zenith_idc])
        o4_damf_norm = o4_damf[zenith_idc] - self.thresholds['O4_TH']
        o4_damf_norm = o4_damf_norm[sza_mask & cloud_mask & finite_mask]

        norm_bin_range = tuple(self.config['normalize_o4_range'])
        norm_bin_size = self.config['normalize_o4_bin_size']
        norm_bins = np.arange(*norm_bin_range, norm_bin_size)
        center_bins = 0.5 * (norm_bins[:-1] + norm_bins[1:])
        count_abs, _ = np.histogram(o4_damf_norm, norm_bins)
        count_rel = count_abs / np.sum(count_abs)

        # Start values for fit
        offset = 0
        scale = np.nanmax(count_rel)
        mu = center_bins[np.argmax(count_rel)]
        sigma = 0.2

        fit_params, _ = curve_fit(gauss, center_bins, count_rel,
                                  [offset, scale, mu, sigma])
        normalization_o4 = fit_params[2]
        self.config['normalization_o4'] = normalization_o4
        if verbose:
            calib_msg = 'O4 dAMF normalization value is '
            calib_msg += f'{normalization_o4:.3f}'
            self.print_log(calib_msg)
        if plot_stream is not None:
            fit_line = gauss(center_bins, *tuple(fit_params))
            fig, ax = plt.subplots(figsize=[8, 4])
            ax.plot(center_bins, count_rel, c='k', alpha=0.7,
                    label='O4 DAMF distribution')
            ax.plot(center_bins, fit_line, '--', color='r', label='Gauss fit')
            ax.axvline(normalization_o4, color='r',
                       label=f'O4 norm. value\nO4Norm={normalization_o4:.2f}')
            ax.legend(loc='upper right')
            ax.set_xlabel('Difference of measured O$_{4}$ DAMF'
                          'and O$_{4}$ Threshold')
            ax.set_ylabel('Relative frequency')
            if 'site' in self.config:
                ax.annotate(self.config['site'], xy=(0.02, 0.9),
                            xycoords='axes fraction')
            ax.grid(alpha=0.4)
            plt.tight_layout()
            plot_stream.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close()
        return normalization_o4

    def calc_spread(self, values: np.array,
                    normalize_sza_dependence: bool=True):
        '''
        Derive the spread of an array (maximum value - minimum value) with 2
        dimensions: scan, elevation
        Spread is derived for each scan.

        Requires gen_thresholds(...) to be run before if
        normalize_sza_dependence is True.

        Data variables:
        @values
        All data variables are required to have the same shape/size

        @normalize_sza_dependence (default=True) is set True if the derived
            spread should be normalized by AVG_SZA values

        RETURNS 1D-array with spread values
        '''
        # Sometimes a scan only contains nan values. This would create a
        # warning that can be ignored.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spread = (np.nanmax(values, axis=1) - np.nanmin(values, axis=1))
        # Normalize spread for SZA dependence
        if normalize_sza_dependence:
            spread /= self.thresholds['AVG_SZA']
        return spread

    def calc_tsi(self, elev: np.array, ci: np.array, dt: np.array,
                 lon: np.array, select_elev: float=None,
                 definition: str='2024_validation'):
        '''
        Derive the temporal smoothness indicator (TSI) for a selected elevation
        angle

        Data variables:
        @elev measured elevation angle
        @ci color index of radiance at 330nm / radiance at 390nm
        @dt datetime information of the measurement
        @lon longitude of the instrument location
        All data variables are required to have the same shape/size
        @select_elev (default=self.config['zenith_elevation'])
        @definition (default='2024_validation') there are different definitions
          to calculate the TSI:
          - 2016_paper: abs(ci_(i-1) + ci_(i+1) / 2 - ci_i)
          - 2024_validation: abs(ci_(i) - ci(i-1))

        RETURNS array of temporal smoothness indicator (TSI) with the same
            shape as the given color index data variable (CI)
        '''
        if select_elev is None:
            select_elev = self.config['zenith_elevation']
        self.check_thresholds()
        self.check_ci_normalization()
        self.check_shape(elev, ci, dt, lon)

        daily_idc = get_daily_idc_before_after(dt, elev, lon, select_elev)

        tsi_scaled = np.ma.copy(ci)
        tsi_scaled.mask = True
        ci_scaled = ci / self.config['normalization_ci']
        ci_scaled /= self.thresholds['CI_AOD02'][:, None]
        for _, cur_daily_idc in daily_idc.items():
            cur_idc = cur_daily_idc['cur_idc']
            prev_idc = cur_daily_idc['prev_idc']
            next_idc = cur_daily_idc['next_idc']
            if cur_idc is None:
                continue
            if definition == '2016_paper':
                tmp = ci_scaled[prev_idc]
                tmp += ci_scaled[next_idc]
                tmp = np.ma.abs(tmp * 0.5 - ci_scaled[cur_idc])
            elif definition == '2024_validation':
                invalid = (cur_idc[0] - prev_idc[0]) > 1
                tmp = np.ma.abs(ci_scaled[cur_idc] -
                                ci_scaled[prev_idc])
                tmp[invalid] = np.nan
            tsi_scaled[cur_idc] = tmp
        return tsi_scaled

    def classify_ci_cloud(self, elev: np.array, ci: np.array,
                          dt: np.array, lon: np.array):
        '''
        Perform cloud classification for type 1 to 5 (see Wagner et al., 2014),
        purely based on color index of zenith measurements.

        Requires gen_thresholds(...) to be run before.

        Data variables:
        @dt datetime information of the measurement
        @ci color index of radiance at 330nm / radiance at 390nm
        @elev measured elevation angle
        @lon longitude of the instrument location
        All data variables are required to have the same shape/size

        RETURNS MDCloudResult object (see toolbox/md_cloud_result.py)
        '''
        self.check_ci_normalization()
        self.check_thresholds()
        self.check_classification_mask()
        self.check_shape(elev, ci, dt, lon)
        # Primary cloud classification using zenith measurements
        nscans = elev.shape[0]
        zenith_idc = get_idc(elev, self.config['zenith_elevation'],
                             order='last')
        # Apply the classification mask
        ci[self.classification_mask] *= np.nan
        ci_scaled = ci[zenith_idc] / self.config['normalization_ci']
        # Calculate the spread only for those scans that contain
        # a zenith measurement
        ci_spread = self.calc_spread(ci[zenith_idc[0]] /
                                     self.config['normalization_ci'])
        # Derive TSI for all zenith measurements
        zenith_tsi = self.calc_tsi(elev, ci, dt, lon,
                                   self.config['zenith_elevation'])
        # Select the TSI of the last zenith measurement in a sequence
        # This corresponds to the TSI as defined in FRM4DOAS D2.2 ATBD
        zenith_tsi = zenith_tsi[zenith_idc]

        # cloud classification
        cloud_type = MDCloudResult(nscans)
        # Flag 1: clear sky low aerosol (main category)
        main1 = ci_scaled >= self.thresholds['CI_TH']
        main1 &= zenith_tsi < self.thresholds['TSI_TH']
        main1 = fill_masked_array(main1)
        cloud_type.main[main1, 0] = 1
        # Flag 2: clear sky high aerosol (main category)
        main2 = ci_scaled < self.thresholds['CI_TH']
        main2 &= zenith_tsi < self.thresholds['TSI_TH']
        main2 &= ci_spread >= self.thresholds['SPREAD_CI']
        main2 = fill_masked_array(main2)
        cloud_type.main[main2, 1] = 1
        # Flag 3: cloud holes (main category)
        main3 = ci_scaled >= self.thresholds['CI_TH']
        main3 &= zenith_tsi >= self.thresholds['TSI_TH']
        main3 = fill_masked_array(main3)
        cloud_type.main[main3, 2] = 1
        # Flag 4: broken clouds (main category)
        main4 = ci_scaled < self.thresholds['CI_TH']
        main4 &= zenith_tsi >= self.thresholds['TSI_TH']
        main4 = fill_masked_array(main4)
        cloud_type.main[main4, 3] = 1
        # Flag 5: continuous clouds (main category)
        main5 = ci_scaled < self.thresholds['CI_TH']
        main5 &= zenith_tsi < self.thresholds['TSI_TH']
        main5 &= ci_spread < self.thresholds['SPREAD_CI']
        main5 = fill_masked_array(main5)
        cloud_type.main[main5, 4] = 1
        # Flag 6: constantly clear (sub category)
        sub1 = main1 | main2
        sub1 &= zenith_tsi < self.thresholds['TSI_CONST_TH']
        sub1 = fill_masked_array(sub1)
        cloud_type.sub[sub1, 0] = 1
        # Flag 7: constantly cloudy (sub category)
        sub2 = main5 & zenith_tsi < self.thresholds['TSI_CONST_TH']
        sub2 = fill_masked_array(sub2)
        cloud_type.sub[sub2, 1] = 1
        return cloud_type

    def classify_o4_cloud(self, elev: np.array, ci: np.array,
                          o4_damf: np.array, cloud_type: MDCloudResult):
        '''
        Perform cloud classification for type 6 and 7 (thick clouds) using the
        o4_damf retrieved from zenith measurements (see Wagner et al., 2016).

        Requires gen_thresholds(...) to be run before.

        Data variables:
        @elev measured elevation angle
        @ci color index of radiance at 330nm / radiance at 390nm
        @o4_damf retrieved O4_DAMF = O4_DSCD / O4_VCD
        All data variables are required to have the same shape/size
        @cloud_type MDCloudResult object (see toolbox/md_cloud_result.py)

        RETURNS MDCloudResult object (see toolbox/md_cloud_result.py)
        '''
        self.check_o4_normalization()
        self.check_thresholds()
        self.check_classification_mask()
        self.check_shape(elev, ci, o4_damf)
        # secondary cloud classification
        zenith_idc = get_idc(elev, self.config['zenith_elevation'],
                             order='last')
        o4_damf[self.classification_mask] *= np.nan
        # calculate the spread only for those scans that contain a
        # zenith measurement
        spread_o4 = self.calc_spread(o4_damf[zenith_idc[0]])
        ci_scaled = ci[zenith_idc] / self.config['normalization_ci']
        # Flag 6: fog
        sub3 = ci_scaled < self.thresholds['CI_TH']
        sub3 &= spread_o4 < self.thresholds['SPREAD_O4']
        sub3 = fill_masked_array(sub3)
        cloud_type.sub[sub3, 2] = 1
        # Flag 7: thick clouds
        o4_amf = o4_damf[zenith_idc] - self.config['normalization_o4']
        o4_th = self.thresholds['O4_TH'] + self.thresholds['O4_TH_OFFSET']
        sub4 = cloud_type.main[:, 3] >= 1
        sub4 |= cloud_type.main[:, 4] >= 1
        sub4 &= o4_amf > o4_th
        sub4 = fill_masked_array(sub4)
        cloud_type.sub[sub4, 3] = 1
        return cloud_type

    def get_warning_flags(self, elev: np.array, dt: np.array,
                          cloud_type: MDCloudResult):
        '''
        Implementation of additional cloud flags:
          - Change of total cloud flag
          - Classification was not performed
          - Check for two zenith angle measurements within a scan
          - Check for time duration of a scan

        Data variables:
        @elev measured elevation angle
        @dt datetime information of the measurement
        All data variables are required to have the same shape/size
        @cloud_type MDCloudResult object (see toolbox/md_cloud_result.py)

        RETURNS MDCloudResult object (see toolbox/md_cloud_result.py)
        '''
        self.check_classification_mask()
        self.check_shape(elev, dt)
        # 1) Change of total cloud flag:
        total_cloud_type = np.append(cloud_type.main, cloud_type.sub, axis=1)
        # Derive number corresponding the the n-bit cloud flag
        total_class = np.packbits(total_cloud_type, axis=1).view(np.uint16)
        total_class = total_class.flatten()
        # Check for changes FRM4DOAS D2.2 ATBD section 5.3
        # Compare previous, current and next scan
        warn1 = total_class[:-2] != total_class[1:-1]
        warn1 |= total_class[1:-1] != total_class[2:]
        warn1_first = total_class[0] != total_class[1]
        warn1_last = total_class[-2] != total_class[-1]
        warn1 = np.insert(warn1, [0, -1], [warn1_first, warn1_last])
        cloud_type.warn[warn1, 0] = 1

        # 2) Mark scans with less than two zenith elevations in the scan
        masked_elev = np.ma.masked_where(self.classification_mask, elev)
        mark_zenith = np.zeros_like(masked_elev)
        mark_zenith[masked_elev == self.config['zenith_elevation']] = 1
        warn2 = np.where(np.sum(mark_zenith, axis=1) != 2)
        cloud_type.warn[warn2, 1] = 1

        # 3) Mark scans with an extraordinary long time difference between
        # start and end of the scan.
        masked_dt = np.ma.masked_where(self.classification_mask, dt)
        vec_ordinal = np.vectorize(tc.get_unix_epoch)
        masked_dt = vec_ordinal(masked_dt)
        dt_spread = self.calc_spread(masked_dt, normalize_sza_dependence=False)
        warn3 = dt_spread > self.config['delta_time_scan']
        warn3 = fill_masked_array(warn3)
        cloud_type.warn[warn3, 2] = 1

        # 4) Mark scans where the main cloud classification was not performed
        warn4 = np.sum(cloud_type.main, axis=1) == 0
        cloud_type.warn[warn4, 3] = 1
        return cloud_type

    def classify_all(self, sza: np.array, elev: np.array, ci: np.array,
                     o4_damf: np.array, dt: np.array, lon: np.array,
                     plot_stream: PdfPages=None, verbose=False):
        '''
        Run full cloud classification. Normalization of color index (ci) and
        O4 DAMF (o4_damf) will only be performed if the respective values
        are not provided in the configuration file.

        Data variables:
        @sza solar zenith angle
        @elev measured elevation angle
        @ci color index of radiance at 330nm / radiance at 390nm
        @o4_damf retrieved O4_DAMF = O4_DSCD / O4_VCD
        @dt datetime information of the measurement
        @lon longitude of the instrument location
        All data variables are required to have the same shape/size

        @plot_stream (default=None) if given, the normalization plots will be
            created on the given PdfPages file stream
        @verbose (default=False) if cloud classification output should be
            printed

        RETURNS MDCloudResult object (see toolbox/md_cloud_result.py)
        '''
        self.check_shape(sza, elev, ci, o4_damf, dt, lon)
        self.gen_thresholds(sza, elev)
        self.set_classification_mask(sza, elev)
        self.normalize_ci(sza, elev, ci, plot_stream=plot_stream,
                          verbose=verbose)
        cloud_type = self.classify_ci_cloud(elev, ci, dt, lon)
        self.normalize_o4(sza, elev, o4_damf, cloud_type,
                          plot_stream=plot_stream, verbose=verbose)
        cloud_type = self.classify_o4_cloud(elev, ci, o4_damf, cloud_type)
        cloud_type = self.get_warning_flags(elev, dt, cloud_type)
        return cloud_type
