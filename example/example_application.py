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
from toolbox import file_tools


def frm4doasdate_to_datetime(fracday, year):
    if year > 10000 or not np.isfinite(fracday):
        # Invalid year: 65535
        # Invalid day: -1
        return None
    return datetime(year, 1, 1) + timedelta(days=fracday - 1)


def mapancdate_to_datetime(longdate):
    if longdate == -1:
        # Invalid year: 65535
        # Invalid day: -1
        return None
    return datetime.strptime(f'{longdate:14d}', '%Y%m%d%H%M%S')


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


def load_mapa_converter_data(mapa_conv_fn):
    vec_dt = np.vectorize(mapancdate_to_datetime)
    data = {}
    # Little specialities in the validation data set
    if 'Dory' in mapa_conv_fn:
        o4_scd_name = 'o4fix_fw352to387nm'
    else:
        o4_scd_name = 'o4_fw352to387nm'
    if 'BREMEN' in mapa_conv_fn:
        rad390_name = 'rad_387'
    else:
        rad390_name = 'rad_390'
    with Dataset(mapa_conv_fn, 'r') as ncin:
        data['sza'] = ncin['measurement_info/sza'][:]
        data['elev'] = ncin['measurement_info/elevation_angle'][:]
        data['rad330'] = ncin['radiance/rad_330/value'][:]
        data['rad390'] = ncin[f'radiance/{rad390_name}/value'][:]
        data['dtraw'] = ncin['measurement_info/time'][:]
        data['o4_scd'] = ncin[f'aerosol/{o4_scd_name}/value'][:]
        data['o4_vcd'] = ncin['auxiliary/o4vcd'][:]
        data['lon'] = ncin['instrument_location/longitude'][:]
    data['dt'] = vec_dt(data['dtraw'])
    data['ci'] = data['rad330'] / data['rad390']
    data['o4_damf'] = data['o4_scd'] / data['o4_vcd'][:, None]
    data['elev'] = np.tile(data['elev'], (data['sza'].shape[0], 1))
    data['lon'] = np.tile(data['lon'], (data['sza'].shape[1], 1)).transpose()
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
data = load_mapa_converter_data(data_fns[0])

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
    # Alternatively call (all steps are performed as above):
    # cloud_class.classify_all(data['sza'], data['elev'], data['ci'],
    #                          data['o4_damf'], data['dt'], data['lon'],
    #                          pdfout, verbose=True)
print(cloud_type)
