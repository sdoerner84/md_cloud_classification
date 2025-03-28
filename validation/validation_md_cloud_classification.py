'''
Created on 07.01.2025

@author: steffen.ziegler
'''
from datetime import datetime
import numpy as np
import yaml
from netCDF4 import Dataset
import md_cloud_classification as mcc
from matplotlib.backends.backend_pdf import PdfPages
from md_cloud_classification.toolbox import file_tools
from validation.cloud_classification_mapa import CloudClassification


def mapancdate_to_datetime(longdate):
    if longdate == -1:
        # Invalid year: 65535
        # Invalid day: -1
        return None
    return datetime.strptime(f'{longdate:14d}', '%Y%m%d%H%M%S')


def load_mapa_converter_data(mapa_conv_fn):
    vec_dt = np.vectorize(mapancdate_to_datetime)
    data = {}
    if 'MAINZ' in mapa_conv_fn:
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


station_name = 'BREMEN'
general_cfg_fn = 'v1_config.yml'
threshold_cfg_fn = 'v1_thresholds.yml'

with open(general_cfg_fn, 'r') as cfg:
    config = yaml.load(cfg, Loader=yaml.SafeLoader)

with open(threshold_cfg_fn, 'r') as cfg:
    thresholds_config = yaml.load(cfg, Loader=yaml.SafeLoader)

datadir = r'D:\sziegler\studies\sziegler\cloud_classification\code_von_lucas\data'
data_fns = file_tools.get_filelist(datadir, recursive=True,
                                   must_contain=[station_name])
data = load_mapa_converter_data(data_fns[0])

# Folder paths
v0_config_fn = 'v0_config_frm4doas_stud.yml'
v0_threshold_fn = 'v0_config_threshold.yml'
if 'MAINZ' in station_name:
    o4_scd_name = 'o4fix_fw352to387nm'
else:
    o4_scd_name = 'o4_fw352to387nm'
if station_name == 'MAINZ':
    calibration_CI_scale = 0.9090909090909091
elif station_name == 'BREMEN':
    calibration_CI_scale = 1.1627906976744187
elif station_name == 'CABAUW':
    calibration_CI_scale = 1.5151515151515151
elif station_name == 'THESSALONIKI':
    calibration_CI_scale = 0.9433962264150942
elif station_name == 'UCCLE':
    calibration_CI_scale = 1.1363636363636365
cc_now = CloudClassification(data_fns, config_data=v0_config_fn,
                             threshold_data=v0_threshold_fn,
                             site=station_name, SZAlim=75,
                             calibration_CI_scale=calibration_CI_scale,
                             O4_dSCD_MAPA=o4_scd_name)
cc_now.TH = cc_now.gen_thresholds()
cc_now.calc_frac()
v0_ci = cc_now.CI[:, cc_now.idx_zenith]
v0_spread_ci = cc_now.calc_spread(cc_now.CI, cc_now.calibration_CI_scale)
config['normalization_ci'] = 1 / calibration_CI_scale
config['normalization_o4'] = -1 * cc_now.calibration_O4_FRS
cloud_class = mcc.MAXDOASCloudClassification(config, thresholds_config)

with PdfPages('D:\\test.pdf') as pdfout:
    cloud_class.check_shape(data['sza'], data['elev'], data['ci'],
                            data['o4_damf'], data['dt'], data['lon'])
    TH = cloud_class.gen_thresholds(data['sza'], data['elev'])
    cloud_class.set_classification_mask(data['sza'], data['elev'])
    zenith_idc = mcc.get_idc(data['elev'], config['zenith_elevation'],
                             order='last')
    v1_ci = np.copy(data['ci'])
    v1_ci[cloud_class.classification_mask] *= np.nan
    v1_ci = v1_ci[zenith_idc]
    cloud_class.normalize_ci(data['sza'], data['elev'], data['ci'],
                             plot_stream=pdfout)
    cloud_type = cloud_class.classify_ci_cloud(data['elev'], data['ci'],
                                               data['dt'], data['lon'])
    v2_spread_ci = cloud_class.calc_spread(data['ci'][zenith_idc[0]] /
                                           cloud_class.config['normalization_ci'])
    v21_spread_ci = cloud_class.calc_spread(data['ci'][zenith_idc[0]]) / cloud_class.config['normalization_ci']
    cloud_class.normalize_o4(data['sza'], data['elev'], data['o4_damf'],
                             cloud_type, plot_stream=pdfout)
    cloud_type = cloud_class.classify_o4_cloud(data['elev'], data['ci'],
                                               data['o4_damf'], cloud_type)
    cloud_type = cloud_class.get_warning_flags(data['elev'], data['dt'],
                                               cloud_type)
print(cloud_type)
