'''
Created on 05.02.2025

@author: steffen.ziegler
'''
from datetime import timedelta
import numpy as np
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, dates as mdates


class MDCloudResult():
    '''
    Class that holds all results of the MAX-DOAS cloud classification
    algorithm.

    Comments behind each label refer to the definition of those categories in
    Wagner et al., 2014 (https://doi.org/10.5194/amt-7-1289-2014)
    Wagner et al., 2016 (https://doi.org/10.5194/amt-9-4803-2016)
    Wagner et al., 2024 FRM4DOAS D2.2 ATBD
    '''

    def __init__(self, nscans):
        '''
        Initial result arrays with number of scans in the data set.
        @nscans
        '''
        self.nscans = nscans
        self.keys = ['main', 'sub', 'warn']
        # Main categories
        self.main = {}
        self.main['long_name'] = 'Main categories (mutually exclusive)'
        self.main['labels'] = ['clear sky low aerosol',   # index 0 - type 1
                               'clear sky high aerosol',  # index 1 - type 2
                               'cloud holes',             # index 2 - type 3
                               'broken clouds',           # index 3 - type 4
                               'continuous',              # index 4 - type 5
                               'empty',                   # index 5
                               'empty',                   # index 6
                               'empty',                   # index 7
                               ]
        self.main['colors'] = ['#4169E1',  # Clear sky low aerosols
                               '#00CCCC',  # Clear sky high aerosols
                               '#BABABA',  # Cloud holes
                               '#BABABA',  # Broken clouds
                               '#FF0000',  # Continuous clouds
                               '#000000',  # Empty
                               '#000000',  # Empty
                               '#000000',  # Empty
                               ]
        self.main['markers'] = ['s',  # Clear sky low aerosols
                                's',  # Clear sky high aerosols
                                '^',  # Cloud holes
                                'v',  # Broken clouds
                                's',  # Continuous clouds
                                's',  # Empty
                                's',  # Empty
                                's',  # Empty
                                ]
        self.main['values'] = np.zeros((nscans, 8), dtype=int)
        # Sub categories
        self.sub = {}
        self.sub['long_name'] = 'Sub categories'
        self.sub['labels'] = ['constantly clear',          # index 0 - new since ATBD
                              'constantly cloudy',         # index 1 - new since ATBD
                              'fog',                       # index 2 - type 6
                              'optically thick clouds',    # index 3 - type 7
                              'empty',                     # index 4
                              'empty',                     # index 5
                              'empty',                     # index 6
                              'empty',                     # index 7
                              ]
        self.sub['colors'] = ['#4169E1',  # Constantly clear
                              '#FF0000',  # Constantly cloudy
                              '#C0C0C0',  # Fog
                              '#636363',  # Optical thick
                              '#000000',  # Empty
                              '#000000',  # Empty
                              '#000000',  # Empty
                              '#000000',  # Empty
                              ]
        self.sub['markers'] = ['o',  # Constantly clear
                               'o',  # Constantly cloudy
                               's',  # Fog
                               's',  # Optical thick
                               's',  # Empty
                               's',  # Empty
                               's',  # Empty
                               's',  # Empty
                               ]
        self.sub['values'] = np.zeros((nscans, 8), dtype=int)
        # Warning categories
        self.warn = {}
        self.warn['long_name'] = 'Warnings'
        self.warn['labels'] = ['classification change flag',   # index 0
                               'less than two zenith measurements '
                               'in a scan',                    # index 1
                               'long scan time',               # index 2
                               'no cloud classification',      # index 3
                               'empty',                        # index 4
                               'empty',                        # index 5
                               'empty',                        # index 6
                               'empty',                        # index 7
                               ]
        self.warn['colors'] = ['#FF9900',  # index 0
                               '#FFE0B3',  # index 1
                               '#995C00',  # index 2
                               '#000000',  # index 3
                               '#000000',  # index 4
                               '#000000',  # index 5
                               '#000000',  # index 6
                               '#000000',  # index 7
                               ]
        self.warn['markers'] = ['<',  # index 0
                                '^',  # index 1
                                'v',  # index 2
                                's',  # index 3
                                's',  # index 4
                                's',  # index 5
                                's',  # index 6
                                's',  # index 7
                                ]
        self.warn['values'] = np.zeros((nscans, 8), dtype=int)

    def __str__(self):
        msg = f"Fraction of all scans (N = {self.nscans}):\n\n"
        for key in self.keys:
            flag = getattr(self, key)
            cloud_abs = np.sum(flag['values'], axis=0)
            cloud_frac = cloud_abs * 100 / self.nscans
            msg += f"{flag['long_name']}:\n"
            for flag_idx, flag_key in enumerate(flag['labels']):
                if flag_key == 'empty':
                    continue
                msg += f"{flag_key}: {cloud_frac[flag_idx]:.2f} %  "
                msg += f"({cloud_abs[flag_idx]})\n"
            msg += '\n'
        return msg
