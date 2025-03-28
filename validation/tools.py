# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:48:42 2020

@author: Vinod
"""
import numpy as np
import netCDF4
import yaml
from datetime import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class nc():
    def universal_gas_constant():
        return 8.3144621    # J / (K * mol)

    def Avogadros_number():
        return 6.022140857E23

    def msl_pressure():
        return 1.01325E5  # Pa

    def scale_height():
        return 7400   # meters for P0 = 101325 Pa and 250K


def return_exp(x, base):
    '''
    Returns the exponent of a base
    '''
    i = 0
    while x >= base:
        x /= base
        i += 1
    return i


def O4_conc(P, T):
    '''
    calculate O4 concentration
    if temperature and pressure are provided
    P: Pressure (Pa)
    T: temperature in K
    '''
    P = np.asarray([P], dtype=float) if np.isscalar(P) else np.asarray(P)
    T = np.asarray([T], dtype=float) if np.isscalar(T) else np.asarray(T)
    # check temperatire units
    if np.nanmean(T < 60):
        T += 273.15
    R = nc.universal_gas_constant()
    N_A = nc.Avogadros_number()
    n_air = (P*N_A)/(T*R)  # molecules m-3
    n_air *= 1E-6
    c_o2 = 0.20942
    n_o4 = (c_o2*n_air)**2
    n_o4 = np.asscalar(n_o4) if len(n_o4) == 1 else n_o4
    # unit molecules2cm-5
    return n_o4


def O4_VCD_sfc(sfc_temp, sfc_height):    # sfc temp in Kelvin, SFC_height in m
    '''
    calculate O4 VCD assuming lapse rate of -0.65k per 100m
    if surface temp and sfc height is provided
    '''
    h = np.arange(sfc_height, 20000, 100)
    temp_profile = [sfc_temp+i*(-0.65) if j <= 12000 else 215
                    for i, j in enumerate(h)]
    pressure_profile = nc.msl_pressure()*np.exp((-h)*temp_profile/nc.scale_height()/250)
    O2_conc = vmr2conc(0.20946*1E9, pressure_profile, temp_profile)
    O4_VCD = np.sum(100*100*O2_conc**2)  # box height 100m= 100*100cm
    return O4_VCD


def conc2vmr(conc, P, T):
    # returns VMR in ppb from concentration in molecules cm-3
    P = np.asarray([P], dtype=float) if np.isscalar(P) else np.asarray(P)
    T = np.asarray([T], dtype=float) if np.isscalar(T) else np.asarray(T)
    N_A = nc.Avogadros_number()
    if np.nanmean(T < 60):
        T += 273.15
    vmr = conc*(8.314*T*1E6)
    vmr /= N_A*1E-9*97000
    vmr = np.asscalar(vmr) if len(vmr) == 1 else vmr
    return vmr


def vmr2conc(vmr, P, T):
    P = np.asarray([P], dtype=float) if np.isscalar(P) else np.asarray(P)
    T = np.asarray([T], dtype=float) if np.isscalar(T) else np.asarray(T)
    N_A = nc.Avogadros_number()
    conc = vmr*1E-9*N_A*P*1E-6/nc.universal_gas_constant()/T
    conc = np.asscalar(conc) if len(conc) == 1 else conc
    return conc


class tracer_prop():
    def __init__(self, tracer):
        self.tracer = tracer
        self.name = tracer['name']
        self.fit_name = tracer['fit_name']
        self.qdoas_name = tracer['qdoas_name']
        self.rms_th = float(tracer['rms_th'])
        self.mapa_spec = tracer['mapa_spec']
        self.dSCD_lim = {key: [float(i) for i in val]
                         for key, val in tracer['dSCD_lim'].items()}
        self.vcd_lim = [float(i) for i in tracer['vcd_lim']]
        self.conc_lim = [float(i) for i in tracer['conc_lim']]
        self.vmr_lim = [float(i) for i in tracer['vmr_lim']]
        self.vcd_label = tracer['vcd_label']
        self.dSCD_label = tracer['dSCD_label']
        self.conc_label = tracer['conc_label']
        self.vmr_label = tracer['vmr_label']
        self.mol_wt = float(tracer['mol_wt'])


def get_tracerdata(tracer_name):
    with open('configuration/config_tracer.yml') as conf:
        tracer_prop_dict = yaml.safe_load(conf)
    return tracer_prop(tracer_prop_dict[tracer_name])


def qdoas_loader(qdoas_file, **kwargs):
    clean_data = kwargs.get('clean_data', True)
    skip_rows = kwargs.get('skip_rows', 0)
    fill_val = kwargs.get('fill_val', 9.9692e+36)

    data = pd.read_csv(qdoas_file, dtype=None, delimiter='\t',
                       skiprows=skip_rows)
    data = data.iloc[:, :-1]
    data['Date_time'] = data['Date (DD/MM/YYYY)'] + data['Time (hh:mm:ss)']
    data['Date_time'] = pd.to_datetime(data['Date_time'],
                                       format='%d/%m/%Y%H:%M:%S')
    data = data.replace(fill_val, np.nan)
    if clean_data:
        for tracer_name in ['no2_vis', 'no2_uv', 'hcho',
                            'bro', 'o4', 'chocho']:
            try:
                tracer = get_tracerdata(tracer_name)
                data.loc[data[tracer.fit_name+'.RMS'] > tracer.rms_th,
                         tracer.qdoas_name] = np.nan
                data.loc[data['SZA'] > 85, tracer.qdoas_name] = np.nan
            except KeyError:
                print(tracer.qdoas_name + 'not found in analysis result')
    return data


def datetime2time(datetime):
    '''
    convert datetime.datetime to time such that date is 01-01-1900
    '''
    time = datetime.replace(year=1900, month=1, day=1)
    return time


def plot_cloud_ts(cloud_inp, **kwargs):
    ms = kwargs.get('marker_size', 90)
    width = 60
    height = 30
    df = pd.read_csv(cloud_inp, delimiter=',', escapechar='#')
    df.columns = df.columns.str.strip()
    df['Date_time'] = pd.to_datetime(df['Date_time'].apply(str),
                                     format='%Y-%m-%d %H:%M:%S')
    df['date'] = df['Date_time'].dt.date
    df['time'] = df.apply(lambda x: datetime2time(x['Date_time']), axis=1)
    cmap = mpl.colors.ListedColormap(['b', 'goldenrod', 'plum', 'teal', 'k'])
    norm = mpl.colors.BoundaryNorm(np.arange(1, 6), cmap.N)
    verts = list(zip([-width, width, width, -width],
                     [-height, -height, height, height]))
    fig, ax = plt.subplots(figsize=[8, 5])
    scat = ax.scatter(df['date'], df['time'], marker=verts, s=ms,
                      c=df['cloud_type'], cmap=cmap, label=None,
                      vmin=1, vmax=5)
    if len(df['cloud_type'].unique()) < 5:
        ax2 = ax.twinx()
        scat = ax2.scatter(np.zeros(5), np.zeros(5), marker=verts,
                           s=ms, c=1+np.arange(5), cmap=cmap, label=None)
        ax2.get_yaxis().set_visible(False)
    scat2 = ax.scatter(df[df['Thick clouds_o4'] == 1]['date'],
                       df[df['Thick clouds_o4'] == 1]['time'], marker=verts,
                       s=ms, c='r', label='Thick\nclouds')
    scat3 = ax.scatter(df[df['fog'] == 1]['date'],
                       df[df['fog'] == 1]['time'], marker=verts,
                       s=ms, c='grey', label='fog')
    cb = plt.colorbar(scat)
    ticks = [1.6, 2.4, 3.2, 4.0, 4.8]
    cb.set_ticks(ticks)
    labels = ['Clear sky \nlow aerosol', 'Clear sky\nhigh\naerosol',
              'Cloud\nholes', 'Broken\nclouds', 'Continuous\nclouds']
    cb.set_ticklabels(labels)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.set_ylabel('Time (UTC)', size=14)
    ax.legend(loc=[1.11, -0.063])
    ax.grid(alpha=0.3)
    return fig, ax


def parse_nc(nc_obj):
    """
    Original code from Steffen Beirle
    Iterative parsing of netcdf object through all group levels.

    """
    dct = {key: np.array(nc_obj[key][:]) for key in nc_obj.variables}
    for group in nc_obj.groups:
        dct[group] = parse_nc(nc_obj[group])
    nc_attrs = nc_obj.ncattrs()
    for key in nc_attrs:
        dct[key] = nc_obj.getncattr(key)
    return dct


def nc2dict(fname):
    """
    Read a nc file into a dictionary.

    All data variables are read.
    """
    with netCDF4.Dataset(fname, "r") as ncin:
        dct = parse_nc(ncin)
    return(dct)


def reorder_by_sequence(dictionary, substances):
    """
    Reorders and organizes species data based on sequence IDs and elevation
    angles.

    This function processes data from a dictionary containing RADIANCE and QDOAS
    results, organizes the data by unique sequence IDs and unique elevation
    angles, and outputs a new dictionary with the processed data.

    Args:
        dictionary (dict): A dictionary containing the raw input data.
            Expected keys are:
            - 'RADIANCE': Contains 'GEODATA' with keys 'sequence_id'
                          and 'viewing_elevation_angle'.
            - 'QDOAS Results': Contains species data to be processed.
        substances (dict): A dictionary mapping species names to their
                           respective data paths within the 'QDOAS Results'
                           section of the input dictionary.
    """
    # Extract sequence IDs and viewing elevation angles from the dictionary
    sequence_ids = dictionary['RADIANCE']['GEODATA']['sequence_id']
    elevations = dictionary['RADIANCE']['GEODATA']['viewing_elevation_angle']

    # Get the unique sequence IDs and sort them, excluding the first value
    unique_seq_ids = np.unique(sequence_ids)
    # Skip the first id (invalid -1)
    unique_seq_ids = np.sort(unique_seq_ids)[1:]

    # Get unique elevation angles
    unique_elevs = np.unique(elevations)

    data = {}
    # Create an array to store elevation values.
    data['elev'] = np.full((len(unique_seq_ids), len(unique_elevs) + 1), np.nan)

    # Loop through the substances dictionary, which contains species information
    for nkey, (species_name, var) in enumerate(substances.items()):
        # Extract the corresponding species data from the dictionary
        # based on how many levels of keys are needed
        if len(var) != 1:
            species = dictionary['QDOAS Results'][var[0]][var[1]]
        else:
            species = dictionary['QDOAS Results'][var[0]]

        # If dealing with time data, initialize a specific object array for it
        if species_name == 'time':
            data[species_name] = np.full((len(unique_seq_ids),
                                          len(unique_elevs) + 1), np.nan, dtype=object)
        else:
            # Otherwise, initialize a numerical array for other species
            data[species_name] = np.full((len(unique_seq_ids),
                                          len(unique_elevs) + 1), np.nan)
        # Loop through each unique sequence ID
        for idx, sid in enumerate(unique_seq_ids):
            # Find indices in sequence_ids that match the current unique sequence ID
            filter_idx = np.where(sequence_ids == sid)[0]

            # Determine the position of the elevations for these filtered indices
            elev_pos = np.searchsorted(unique_elevs, elevations[filter_idx])

            # For the first species fill the elevation data
            if nkey == 0:
                # Assign elevation data based on the filtered indices
                data['elev'][idx, elev_pos] = elevations[filter_idx]
                # Roll the array to shift elements to the right
                # to sort like zenith, elevations, zenith
                data['elev'][idx] = np.roll(data['elev'][idx], 1)

            # If dealing with time, convert the time format once
            if species_name == 'time' and idx == 0:
                species = convert_timeformat(species)

            # Assign species data to the appropriate positions in the array
            data[species_name][idx, elev_pos] = species[filter_idx]
            data[species_name][idx] = np.roll(data[species_name][idx], 1)

            try:
                # Assign the minimum index value to the first
                # position of the species array
                data[species_name][idx][0] = species[np.nanmin(filter_idx)]
                # Also assign the corresponding elevation for the first species
                if nkey == 0:
                    data['elev'][idx][0] = elevations[np.nanmin(filter_idx)]
            except Exception as e:
                print(f"Error with species_name {species_name}"
                      + "and index {idx}: {e}")
                break

    return data


def convert_timeformat(date_time_array):
    result = []
    for index, row in enumerate(date_time_array):
        year = str(row[0]).zfill(4)
        month = str(row[1]).zfill(2)
        day = str(row[2]).zfill(2)
        hour = str(row[3]).zfill(2)
        minute = str(row[4]).zfill(2)
        second = str(row[5]).zfill(2)
        datetime_str = f'{year}{month}{day}{hour}{minute}{second}'
        result.append(datetime_str)
    return np.array(result)


def get_elevation(nc_filepath):
    with netCDF4.Dataset(nc_filepath, 'r+') as nc_file:
        try:
            geodata = nc_file.groups['RADIANCE'].groups['GEODATA']
            elev = geodata.variables['viewing_elevation_angle'][:]
            return np.array(elev)
        except KeyError:
            raise KeyError("Variable 'viewing_elevation_angle'"
                           + "not found in the NetCDF file.")


def create_sequences(elev, zenith=90):
    """
    Create sequences of elevation angles starting and ending
    with zenith measurements.

    This function defines sequences from an array of elevation angles,
    ensuring that each sequence starts and ends with a zenith measurement.
    A zenith measurement can be part of two consecutive sequences.

    Parameters:
    elev (numpy array): Array of elevation angles.
    zenith (float): The zenith measurement value. Defaults to 90.

    Returns:
    numpy array of float with shape (n,2):
    An array of sequence allocated to each elevation angle.

    Example:
    np.array([90,30, 90, 30,])
    [[-1  0]
     [ 0 -1]
     [ 0 -1]
     [-1 -1]]
    """
    # Get index of zenith indices
    zenith_idc = np.where(elev == zenith)[0]
    # Create an (n, 2) array, where n is the length of elevation, containing -1
    # -1 is used as fill value
    seq = np.zeros((elev.size, 2), dtype=int) - 1
    # Looping through each zenith indices
    for seq_number in range(len(zenith_idc)-1):
        # Attribute elevation to sequence between two zenith measurement.
        # Get first zenith index
        start_zenith = zenith_idc[seq_number]
        # Get next zenith index
        stop_zenith = zenith_idc[seq_number + 1]
        # Every elevation between start_zenith and stop_zenith measurement
        # get the same sequence number in first position
        # and -1 in second (as fill value, due to array definition)
        seq[start_zenith + 1:stop_zenith, 0] = seq_number
        # Zenith measurements are assigned to the current (1, begin of sequence)
        seq[start_zenith, 1] = seq_number
        # and subsequent sequence (0, end of sequence)
        seq[stop_zenith, 0] = seq_number
    return seq


def add_sequence_id_variable(nc_filepath, sequences):
    """
    Add or update the 'sequence_id' variable in the NetCDF file.

    Parameters:
    nc_filepath (str): Path to the NetCDF file.
    sequences (np.ndarray): Array of sequence identifiers.

    Raises:
    FileNotFoundError: If the NetCDF file does not exist.
    """
    try:
        with netCDF4.Dataset(nc_filepath, 'a') as nc_file:
            if 'dim2_size' not in nc_file.dimensions:
                nc_file.createDimension('dim2_size', 2)
            if 'sequence_id' not in nc_file['/RADIANCE/GEODATA'].variables:
                dimensions = ('number_of_records', 'dim2_size')
                nc_file.createVariable('/RADIANCE/GEODATA/sequence_id',
                                       datatype='i4', dimensions=dimensions,
                                       fill_value=-1)
            seq_id_var = nc_file['/RADIANCE/GEODATA/sequence_id']
            try:
                seq_id_var[...] = sequences
            except Exception as e:
                print(f"Error occurred: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"NetCDF file '{nc_filepath}' not found.")


def check_sequence_id(nc_filepath):
    """
    Check if 'sequence_id' variable exists in the NetCDF file.

    Parameters:
    nc_filepath (str): Path to the NetCDF file.

    Returns:
    bool: True if 'sequence_id' exists, False otherwise.
    """
    try:
        with netCDF4.Dataset(nc_filepath, 'r+') as nc_file:
            return 'sequence_id' in nc_file['RADIANCE']['GEODATA'].variables
    except FileNotFoundError:
        return False


def read_frm4doas_qdoas(nc_filepath, zenith):
    """
    Process the NetCDF file by adding sequence IDs based on elevation data and 
    reorder the data based on specified substances.

    Parameters:
    nc_filepath (str): Path to the NetCDF file.
    zenith (float): Zenith for sequence definition

    Returns:
    dict: A dictionary with the reordered data.
    Raises:
    Exception: If an error occurs during processing.
    """
    # Check if 'sequence_id' variable exists if not define sequence ID
    if not check_sequence_id(nc_filepath):
        try:
            # Retrieve elevation data and create sequences
            elevation = get_elevation(nc_filepath)
            sequences = create_sequences(elevation, zenith=zenith)
            add_sequence_id_variable(nc_filepath, sequences)
            print("Variable 'sequence_id' added in the NetCDF file.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while adding sequence ID: {e}")

    data = nc2dict(nc_filepath)

    # Define substances and their corresponding variable names
    substances = {
        'o4_dSCD': ['O4_fix', 'SlCol(O4T293K)'],
        'no2_vis': ['NO2_VIS', 'SlCol(NO2T294K)'],
        'rad330':  ['Fluxes 330'],
        'rad380':  ['Fluxes 380'],
        'rad390':  ['Fluxes 390'],
        'sza':     ['SZA'],
        'time':    ['Date & time (YYYYMMDDhhmmss)']
    }

    try:
        # Reorder the data by the sequence of substances
        data_sorted = reorder_by_sequence(data, substances)
    except Exception as e:
        raise RuntimeError(f"An error occurred while reordering data: {e}")
    
    return data_sorted
