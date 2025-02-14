'''
Created on 05.02.2025

@author: steffen.ziegler
'''
import numpy as np


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
        self.group_labels = {'main': 'Main categories (mutually exclusive)',
                             'sub': 'Sub categories',
                             'warn': 'Warnings'}
        self.main_labels = ['clear sky low aerosol',    # index 0 - type 1
                            'clear sky high aerosol',   # index 1 - type 2
                            'cloud holes',              # index 2 - type 3
                            'broken clouds',            # index 3 - type 4
                            'continuous',               # index 4 - type 5
                            'empty',                    # index 5
                            'empty',                    # index 6
                            'empty',                    # index 7
                            ]
        self.main = np.zeros((nscans, 8), dtype=int)
        self.sub_labels = ['constantly clear',          # index 0 - new since ATBD
                           'constantly cloudy',         # index 1 - new since ATBD
                           'fog',                       # index 2 - type 6
                           'optically thick clouds',    # index 3 - type 7
                           'empty',                     # index 4
                           'empty',                     # index 5
                           'empty',                     # index 6
                           'empty',                     # index 7
                           ]
        self.sub = np.zeros((nscans, 8), dtype=int)
        self.warn_labels = ['classification change flag',   # index 0
                            'less than two zenith measurements '
                            'in a scan',                    # index 1
                            'long scan time flag',          # index 2
                            'no cloud classification',      # index 3
                            'empty',                        # index 4
                            'empty',                        # index 5
                            'empty',                        # index 6
                            'empty',                        # index 7
                            ]
        self.warn = np.zeros((nscans, 8), dtype=int)

    def __str__(self):
        msg = f"Fraction of all scans (N = {self.nscans}):\n\n"
        for group_key, group_desc in self.group_labels.items():
            cloud_abs = np.sum(getattr(self, group_key), axis=0)
            cloud_frac = cloud_abs * 100 / self.nscans
            group_labels = getattr(self, group_key + '_labels')
            msg += f"{group_desc}:\n"
            for flag_idx, flag_key in enumerate(group_labels):
                if flag_key == 'empty':
                    continue
                msg += f"{flag_key}: {cloud_frac[flag_idx]:.2f} %  "
                msg += f"({cloud_abs[flag_idx]})\n"
            msg += '\n'
        return msg
