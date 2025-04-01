'''
Created on 07.01.2025

@author: steffen.ziegler

Required modules (in addition to the modules listed in
mpic_cloud_classification):
pyyaml 6.0.2
netCDF4 1.7.2

This is just an example in order to see which data fields are required and how
the cloud classification algorithm is executed.
'''
from datetime import datetime, timedelta
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from netCDF4 import Dataset
from md_cloud_classification import MAXDOASCloudClassification
from md_cloud_classification.toolbox import file_tools


def frm4doasdate_to_datetime(fracday, year):
    if year > 10000 or not np.isfinite(fracday):
        # Invalid year: 65535
        # Invalid day: -1
        return None
    return datetime(year, 1, 1) + timedelta(days=fracday - 1)


def load_frm4doas_data(frm4doas_fn, o4_vcd=1.237656277109714e43):
    vec_dt = np.vectorize(frm4doasdate_to_datetime)
    data = {}
    with Dataset(frm4doas_fn, 'r') as ncin:
        grp = ncin['DIFFERENTIAL_SLANT_COLUMN']
        data['sza'] = grp['solar_zenith_angle_of_measured_slant_column_density'][:]
        data['elev'] = grp['elevation_angle_of_telescope'][:]
        data['rad330'] = grp['relative_intensity_around_330_nm'][:]
        data['rad390'] = grp['relative_intensity_around_390_nm'][:]
        data['fracday'] = grp['fractional_day_of_measured_slant_column_density'][:]
        data['year'] = grp['year_of_measured_slant_column_density'][:]
        data['o4_scd'] = grp['O4_fix/slant_column_density_of_o4t293k'][:]
        data['lon'] = ncin['INSTRUMENT_LOCATION/longitude'][:]
    data['ci'] = data['rad330'] / data['rad390']
    data['dt'] = vec_dt(data['fracday'], data['year'])
    np.ma.set_fill_value(data['dt'], -1)
    data['o4_amf'] = data['o4_scd'] / o4_vcd
    return data


general_cfg_fn = 'example_config.yml'
threshold_cfg_fn = '../config/default_thresholds.yml'

with open(general_cfg_fn, 'r') as cfg:
    config = yaml.load(cfg, Loader=yaml.SafeLoader)

with open(threshold_cfg_fn, 'r') as cfg:
    thresholds_config = yaml.load(cfg, Loader=yaml.SafeLoader)

calibration_report_fn = r'D:\\test.pdf'

# Provide a dictionary with a valid MAPA converter file here:
datadir = r'D:\sziegler\studies\sziegler\cloud_classification\code_von_lucas\data'
data_fns = file_tools.get_filelist(datadir, recursive=True,
                                   must_contain=['Dory', '.nc'])
data = load_frm4doas_data(data_fns[0])

cloud_class = MAXDOASCloudClassification(config, thresholds_config)
with PdfPages(calibration_report_fn) as pdfout:
    # Full controll which steps should be performed:
    cloud_class.gen_thresholds(data['sza'], data['elev'])
    cloud_class.set_classification_mask(data['sza'], data['elev'])
    cloud_class.normalize_ci(data['sza'], data['elev'], data['ci'],
                             plot_stream=pdfout, verbose=True)
    cloud_type = cloud_class.classify_ci_cloud(data['elev'], data['ci'],
                                               data['dt'], data['lon'])
    cloud_class.normalize_o4(data['sza'], data['elev'], data['o4_damf'],
                             cloud_type, plot_stream=pdfout, verbose=True)
    cloud_type = cloud_class.classify_o4_cloud(data['elev'], data['ci'],
                                               data['o4_damf'], cloud_type)
    cloud_type = cloud_class.get_warning_flags(data['elev'], data['dt'],
                                               cloud_type)

cloud_types = {'main': {'long_name': "First column - clear sky with low aerosols,\n"
                                     "Second column - clear sky with high aerosols,\n"
                                     "Third column - cloud holes,\n"
                                     "Fourth column - broken clouds,\n"
                                     "Fifth column - continuous clouds,\n"
                                     "Sixth column - None\n"
                                     "Seventh column - None\n"
                                     "Eighth column - None",
                        'ncvar': 'CLOUD/main_classification'},
               'sub': {'long_name': "First column - constantly clear,\n"
                                    "Second column - constantly cloudy,\n"
                                    "Third column - fog,\n"
                                    "Fourth column - optically thick clouds,\n"
                                    "Fifth column - None,\n"
                                    "Sixth column - None\n"
                                    "Seventh column - None\n"
                                    "Eighth column - None",
                       'ncvar': 'CLOUD/sub_classification'},
               'warn': {'long_name': "First column - classification change flag,\n"
                                     "Second column - less than two zenith measurements in a scan,\n"
                                     "Third column - long scan time flag,\n"
                                     "Fourth column - no cloud classification,\n"
                                     "Fifth column - None,\n"
                                     "Sixth column - None\n"
                                     "Seventh column - None\n"
                                     "Eighth column - None",
                        'ncvar': 'CLOUD/warn_classification'},
               }
with Dataset(data_fns[0], 'a') as ncout:
    if 'CLOUD' not in ncout.groups:
        if 'dim_8bit' not in ncout.dimensions:
            ncout.createDimension('dim_8bit', 8)
        for key, item in cloud_types.items():
            dim = ('dim_sequences', 'dim_8bit')
            var_cc = ncout.createVariable(item['ncvar'],
                                          datatype='i4',
                                          dimensions=dim,
                                          fill_value=0)
            var_cc.setncattr('long_name', item['long_name'])
    for key, item in cloud_types.items():
        ncvar = ncout[item['ncvar']]
        ncvar[:] = cloud_type.__dict__[key]

