# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:00:34 2020

@author: Vinod
Script for cloud classification
according to T. Wagner et.al., https://www.atmos-meas-tech.net/9/4803/2016/
and
T. Wagner et.al., https://www.atmos-meas-tech.net/7/1289/2014/ for optional
thick cloud identification
"""
import numpy as np
import yaml
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class CloudClassError(Exception):
    '''
    Define generic sd_toolbox exception
    '''
    def __init__(self, value):
        self.value = value
        self.strerror = value

    def __str__(self):
        return self.strerror


def fitFunc(x, A0, A, mu, sigma):
    return A0 + A*np.exp(-((x-mu)**2/(2.*sigma**2)))


class cloud_class():
    def __init__(self, data, **kwargs):
        '''
        Parameters
        ----------
        data : QDOAS analysis data as pandas dataframe
        VCD_O4 : scalar or vector of length equal to data
            Dictionary of polynomical coefficients from Wagner et al 2016
        calibration_CI_scale : Propotionality constant for CI calibration
        Returns
        -------
        Dictionary of arrays of thresholds
        '''
        VCD_O4 = kwargs.get('O4_vcd', 1.3e43)
        self.calibration_CI_scale = kwargs.get('calibration_CI_scale', None)
        config_data = kwargs.get('config_data',
                                 '../configuration/config_data.yml')
        with open(config_data) as conf_file:
            Inp = yaml.safe_load(conf_file)
        with open('../configuration/config_threshold.yml') as conf_file:
            Inp.update(yaml.safe_load(conf_file))
        data['CI'] = data[Inp['Fluxes_330']]/data[Inp['Fluxes_390']]
        data['dAMF_O4'] = data[Inp['O4_dSCD']]/VCD_O4
        data = data[~np.isin(data[Inp['elev_angle']],
                             Inp['ignore_elev'])].reset_index(drop=True)
        self.data = data
        self.data90 = data[data[Inp['elev_angle']] == 90].reset_index(drop=
                                                                      True)
        self.idx_90 = data[data[Inp['elev_angle']] == 90].index.values
        self.TH_dict = {}
        # get number of usable elevation angles
        if Inp['seq'] is None:
            Inp['seq'] = len(data[Inp['elev_angle']].unique())
            Inp['seq'] -= len([i for i in Inp['ignore_elev']
                               if i in data[Inp['elev_angle']].unique()])
        else:
            input('Number of elevation angles are {}. Are you sure?\
                        \nIf yes, press Enter to continue'.format(Inp['seq']))
        self.Inp = Inp
        self.ctype = None

    def gen_thresholds(self):
        if self.TH_dict:
            return self.TH_dict
        else:
            s_norm = self.data90['SZA']/90
            TH_dict = {}
            for th_now in ['CI_CSR', 'CI_TH', 'CI_MIN', 'TSI_TH', 'O4_TH']:
                poly_temp = np.poly1d(self.Inp[th_now])
                TH_dict[th_now] = poly_temp(s_norm)
                if th_now == 'TSI_TH':
                    TH_dict[th_now] *= 0.06
            self.TH_dict = TH_dict
            return TH_dict

    def normalize_CI(self, **kwargs):
        method = kwargs.get("method", "peak_finding")
        # methods optios "Gauus_fit" or "peak_finding"
        if self.calibration_CI_scale is None:
            TH = self.gen_thresholds()
            CI_Norm = self.data90['CI']/TH['CI_MIN']
            n_bins = np.arange(0, 3.02, 0.02)
            # First gauss fit to clip
            count_abs, bins = np.histogram(CI_Norm[self.data90['SZA'] < 60],
                                           n_bins)
            count = count_abs/np.sum(count_abs)
            fig, ax = plt.subplots(figsize=[8, 4])
            ax.plot(bins[:-1], count, c='k', alpha=0.7,
                    label='CI distribution')
            if method == "Gauss_fit":
                p0 = [0, np.max(count), n_bins[np.argmax(count)], 0.2]
                fitParams, fitCovariances = curve_fit(fitFunc, bins[:-1],
                                                      count, p0)
                # Final gauss fit after clipping
                count_, bins = np.histogram(CI_Norm[(self.data90['SZA'] < 60) &
                                          (CI_Norm < fitParams[2]+fitParams[3])], n_bins)
                count_ = count_/np.sum(count_abs)
                fitParams, fitCovariances = curve_fit(fitFunc, bins[:-1],
                                                      count_, p0)
                fit_line = fitFunc(bins[:-1], fitParams[0], fitParams[1],
                                   fitParams[2], fitParams[3])
                calibration_CI_scale = 1/fitParams[2]
                ax.plot(bins[:-1], fit_line, '-', color='r', label='Gauss fit')
            else:
                peaks, _ = find_peaks(count, prominence=0.05*np.max(count))
                ax.scatter(bins[peaks], count[peaks], marker='o', s=24,
                           facecolor="none", edgecolor='b', label='All peaks')
                peaks_new = peaks[count[peaks] > 0.1*np.max(count[peaks])]
                ax.scatter(bins[peaks_new], count[peaks_new], marker='x',
                           c='r', s=24, label='$>$ 0.1 * max')
                ax.axvline(bins[np.min(peaks_new)], color='r',
                           label='CI calib. value')
                calibration_CI_scale = 1/bins[np.min(peaks_new)]
            frac_lower = np.sum(count[bins[:-1] < (1/calibration_CI_scale)])
            CI_message = 'CI scaled calibration value is '
            CI_message += str(calibration_CI_scale) + '\n'
            CI_message += 'Fraction of CI below threshols is '
            CI_message += str(frac_lower)
            print(CI_message)
            ax.legend(loc='upper right')
            ax.set_xlabel('Normalized CI')
            ax.set_ylabel('Relative frequency')
            ax.annotate(self.Inp['site'], xy=(0.02, 0.9),
                        xycoords='axes fraction')
            ax.annotate('Fraction of CI\nbelow threshold: {0:.2f}%'.format(frac_lower*100),
                        xy=(0.02, 0.8), xycoords='axes fraction')
            ax.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()
            self.calibration_CI_scale = calibration_CI_scale
        return self.calibration_CI_scale

    def calc_spread(self, param, scale=1):
        seq = self.Inp['seq']
        spread = np.zeros(len(self.idx_90))
        for i_sequence in np.arange(len(self.idx_90)):
            if i_sequence == 0:
                spread[i_sequence] = (max(self.data[param][0:self.idx_90[i_sequence]+1]) -
                                      min(self.data[param][0:self.idx_90[i_sequence]+1]))*scale
            else:
                if self.idx_90[i_sequence]-self.idx_90[i_sequence-1] < seq:
                    spread[i_sequence] = np.nan
                    # incomplete sequence having zenith
                elif self.idx_90[i_sequence]-self.idx_90[i_sequence-1] > seq:
                    spread[i_sequence] = (max(self.data[param][self.idx_90[i_sequence]-(seq-1):self.idx_90[i_sequence]+1]) -
                          min(self.data[param][self.idx_90[i_sequence]-(seq-1):self.idx_90[i_sequence]+1]))*scale
                             # incomplete sequence having zenith
                else:
                    spread[i_sequence] = (max(self.data[param][self.idx_90[i_sequence-1]+1:self.idx_90[i_sequence]+1]) -
                          min(self.data[param][self.idx_90[i_sequence-1]+1:self.idx_90[i_sequence]+1]))*scale
        return spread

    def classify_cloud(self):
        # primary cloud classification
        TH = self.gen_thresholds()
        # if accumulation point of CI is not provided manually
        if self.calibration_CI_scale is None:
            self.calibration_CI_scale = self.normalize_CI()
        CI_scaled = self.data90['CI'] * self.calibration_CI_scale
        daystr = self.data90['Date_time'].apply(str)
        daystr = daystr.str[0:11]
        days = list(set(daystr))
        days.sort()
        TSI_scaled = np.zeros(len(self.idx_90))
        for day in days:
            idx_day = daystr.index[daystr == day]
            if len(idx_day) > 2:
                TSI_scaled[idx_day[0]] = 0
                TSI_scaled[idx_day[-1]] = 0
                TSI_scaled[idx_day[1:-1]] = abs((CI_scaled[idx_day[:-2]].values +
                         CI_scaled[idx_day[2:]].values)/2-CI_scaled[idx_day[1:-1]].values)
        spread_CI = self.calc_spread('CI', self.calibration_CI_scale)
        # primary cloud classification
        # primary classification: clear sky low aerosols 1, clear sky high
        # aerosols 2, cloud holes 3, broken clouds 4, continuous clouds 5
        ctype = np.zeros((len(self.idx_90), 3))
        idx1 = (CI_scaled >= TH['CI_TH']) & (TSI_scaled < TH['TSI_TH'])
        ctype[idx1, 0] = 1
        idx2 = (CI_scaled < TH['CI_TH']) & (TSI_scaled < TH['TSI_TH'])
        idx2 &= (spread_CI >= 0.14)
        ctype[idx2, 0] = 2
        idx3 = (CI_scaled >= TH['CI_TH']) & (TSI_scaled >= TH['TSI_TH'])
        ctype[idx3, 0] = 3
        idx4 = (CI_scaled < TH['CI_TH']) & (TSI_scaled >= TH['TSI_TH'])
        ctype[idx4, 0] = 4
        idx5 = (CI_scaled < TH['CI_TH']) & (TSI_scaled < TH['TSI_TH'])
        idx5 &= (spread_CI < 0.14)
        ctype[idx5, 0] = 5

        # secondary cloud classification
        spread_O4 = self.calc_spread('dAMF_O4')
        # fog
        idx6 = (CI_scaled < TH['CI_TH']) & (spread_O4 < 0.37)
        ctype[idx6, 1] = 1

        # O4 calibration
        if self.Inp['Thick_cloud_method'] == 'O4':
            idx_o4_calib = (ctype[:, 0] <= 3)
            idx_o4_calib &= (self.data90['SZA'] > 30) & (self.data90['SZA'] < 50)
            data_o4_norm = self.data90['dAMF_O4'] - TH['O4_TH']
            data_o4_norm = data_o4_norm[idx_o4_calib]
            n_bins = np.arange(-3, 3.01, 0.01)
            count, bins = np.histogram(data_o4_norm, n_bins)
            count = count/np.sum(count)
            p0 = [0, np.max(count), n_bins[np.argmax(count)], 0.1]
            fitParams, fitCovariances = curve_fit(fitFunc, bins[:-1], count,
                                                  p0)
            fit_line = fitFunc(bins[:-1], fitParams[0], fitParams[1],
                               fitParams[2], fitParams[3])
            calibration_O4_FRS = np.negative(fitParams[2])
            print('O4 FRS AMF is ' + str(calibration_O4_FRS))
            fig, ax = plt.subplots()
            ax.plot(bins[:-1], count, color='k', alpha=0.7,
                    label='O4 AMF distribution')
            ax.plot(bins[:-1], fit_line, '-', color='r', label='Gauss fit')
            ax.legend(loc='upper right')
            ax.set_xlabel('Difference of measured O$_{4}$ DAMF and O$_{4}$ Threshold')
            ax.set_ylabel('Relative frequency')
            ax.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()
            O4_AMF = self.data90['dAMF_O4'] + calibration_O4_FRS
            # Thick clouds
            idx7 = (ctype[:, 0] >= 4) & (O4_AMF > TH['O4_TH'] + 0.85)
            ctype[idx7, 2] = 1
        self.ctype = ctype
        return ctype

    def calc_frac(self):
        if self.ctype is None:
            self.classify_cloud()
        ctype = self.ctype
        Freq = [len(ctype[:, 0][ctype[:, 0] == i])/len(ctype[:, 0])
                for i in [1, 2, 3, 4, 5]]
        Freq.append(len(ctype[:, 1][ctype[:, 1] == 1])/len(ctype[:, 0]))
        Freq.append(len(ctype[:, 2][ctype[:, 2] == 1])/len(ctype[:, 0]))
        print('Percentage of \n clear sky low aerosols : ' + str(Freq[0]*100)
              + '\n clear sky high aerosol : ' + str(Freq[1]*100)
              + '\n cloud holes : ' + str(Freq[2]*100)
              + '\n broken clouds : ' + str(Freq[3]*100)
              + '\n continuous clouds : ' + str(Freq[4]*100)
              + '\n fog : ' + str(Freq[5]*100)
              + '\n thick clouds : ' + str(Freq[6]*100))

    def save_csv_out(self, savename, dt_col='Date_time'):
        if self.ctype is None:
            self.classify_cloud()
        data_cloud_classified = np.c_[self.data90[dt_col], self.data90['SZA'],
                                      self.ctype[:, 0:3]]
        header = 'Cloud classification results using python tool with\n'
        header += 'CI_scale = ' + str(self.calibration_CI_scale) + '\n'
        header += 'and O4_ref_AMF = ' + str(self.calibration_O4_FRS) + '\n'
        header += "primary classification- clear sky low aerosols: 1," + '\n'
        header += "clear sky high aerosols: 2, cloud holes: 3, " + '\n'
        header += "broken clouds: 4, continuous clouds: 5, " + '\n'
        header += "No classification: -1" + '\n'
        header += 'Date_time, SZA, cloud_type, fog, Thick clouds_o4'
        np.savetxt(savename, data_cloud_classified, delimiter=',',
                   fmt=('%s', '%.3f', '%i', '%i', '%i'), header=header)
