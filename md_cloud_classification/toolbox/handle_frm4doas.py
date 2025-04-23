'''
Created on 03.04.2025

@author: steffen.ziegler
'''
import textwrap
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import yaml
from matplotlib import pyplot as plt, dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from md_cloud_classification.toolbox import file_tools
from md_cloud_classification.toolbox import plot_tools
from md_cloud_classification.toolbox import time_conversions as tc
from md_cloud_classification.toolbox.md_cloud_result import MDCloudResult
from md_cloud_classification import MAXDOASCloudClassification


CLOUD_GROUP = 'CLOUD'


def frm4doasdate_to_datetime(fracday: float, year: int) -> datetime:
    '''
    Derive a datetime object from given fractional day and year.

    @fracday is a fractional day with the first of January, 0h = 1.0
    @year

    RETURNS a datetime object if the date can be parse, None if not
    '''
    if year > 10000 or not np.isfinite(fracday):
        # Invalid year: 65535
        # Invalid day: -1
        return None
    return datetime(year, 1, 1) + timedelta(days=fracday - 1)


def load_frm4doas_data(frm4doas_fn: str, read_cloudflags: bool=False) -> dict:
    '''
    Read all variables required for the cloud classification from FRM4DOAS L15
    data files.

    @frm4doas_fn is the filename of a FRM4DOAS L15 input file
    @read_cloudflags (Default=False) if set true, the function will read the
                     cloud flag information from the netcdf file

    RETURNS a dictionary containing all data needed for running the cloud
            classification tool.
    '''
    vec_dt = np.vectorize(frm4doasdate_to_datetime)
    vec_epoch = np.vectorize(tc.get_unix_epoch)
    result = {}
    with Dataset(frm4doas_fn, 'r') as ncin:
        grp = ncin['DIFFERENTIAL_SLANT_COLUMN']
        result['sza'] = grp['solar_zenith_angle_of_measured_slant_column_density'][:]
        result['elev'] = grp['elevation_angle_of_telescope'][:]
        result['rad330'] = grp['relative_intensity_around_330_nm'][:]
        result['rad390'] = grp['relative_intensity_around_390_nm'][:]
        result['fracday'] = grp['fractional_day_of_measured_slant_column_density'][:]
        result['year'] = grp['year_of_measured_slant_column_density'][:]
        result['o4_dscd'] = grp['O4_fix/slant_column_density_of_o4t293k'][:]
        result['lon'] = ncin['INSTRUMENT_LOCATION/longitude'][:]
        result['lat'] = ncin['INSTRUMENT_LOCATION/latitude'][:]
        header_attrs = ['campaign_name', 'institution', 'station_name']
        for header_attr in header_attrs:
            result[header_attr] = np.array([ncin.getncattr(header_attr)])
        result['filename'] = np.array([frm4doas_fn])
        if read_cloudflags:
            result['main'] = ncin[f'{CLOUD_GROUP}/main_types'][:]
            result['sub'] = ncin[f'{CLOUD_GROUP}/sub_types'][:]
            result['warn'] = ncin[f'{CLOUD_GROUP}/warnings'][:]
        if 'AUXILLARY' in ncin.groups:
            result['o4_vcd'] = ncin['AUXILLARY/o4_vcd'][:]
    result['ci'] = result['rad330'] / result['rad390']
    result['dt'] = vec_dt(result['fracday'], result['year'])
    result['epoch'] = vec_epoch(result['dt'])
    np.ma.set_fill_value(result['dt'], -1)
    if result['lon'].size != result['dt'].size:
        result['lon'] = np.repeat(result['lon'], result['dt'].size).reshape(result['dt'].shape)
        result['lat'] = np.repeat(result['lat'], result['dt'].size).reshape(result['dt'].shape)
    return result


def load_multiple_frm4doas_data(frm4doas_fns: list, read_cloudflags: bool=False) -> dict:
    '''
    This function reads multiple FRM4DOAS L15 data files and appends them to
    form one result dictionary. This only works if the data files are
    compatible, mainly referring to variables and the maximum number of
    elevation angles. This is mainly needed for running the calibration
    routines.

    @frm4doas_fns list of FRM4DOAS L15 data files
    @read_cloudflags see load_frm4doas_data(...)

    RETURNS a dictionary containing all data needed for running the cloud
            classification tool.
    '''
    result = None
    for frm4doas_fn in frm4doas_fns:
        curdata = load_frm4doas_data(frm4doas_fn,
                                     read_cloudflags=read_cloudflags)
        if result is None:
            result = curdata
            continue
        for key, item in result.items():
            result[key] = np.ma.append(item, curdata[key], axis=0)
    return result


def save_cloudtype_to_frm4doas(frm4doas_fn: str, cloudtype: MDCloudResult,
                               group_name=CLOUD_GROUP):
    '''
    Add a netCDF group for saving the cloud type results to a given
    FRM4DOAS file. This will add the group_name group with the following
    variables:
        main_types
        sub_types
        warnings

    For storing the data, a "dim_8bit" dimension is created.

    @frm4doas_fn path of the FRM4DOAS L15 netcdf file
    @cloudtype is the result of cloud classification given as MDCloudResult
               object
    @group_name (default=CLOUD_GROUP) name of group that contains the cloud
                flags
    '''
    with Dataset(frm4doas_fn, 'a') as ncstream:
        if group_name not in ncstream.groups:
            grp = ncstream.createGroup(group_name)
        else:
            grp = ncstream[group_name]
        if 'dim_8bit' not in ncstream.dimensions:
            ncstream.createDimension('dim_8bit', 8)
        for key in cloudtype.keys:
            flag = getattr(cloudtype, key)
            if 'warn' in key:
                ncvarname = 'warnings'
            else:
                ncvarname = key + '_types'
            if ncvarname not in grp.variables:
                # Generate description
                long_name = [flag['long_name']]
                for idx, desc in enumerate(flag['labels']):
                    long_name.append(f"Column {idx}: {desc}")
                long_name = '\n'.join(long_name)
                # Create variable
                dim = ('scan_dimension', 'dim_8bit')
                var_cc = grp.createVariable(ncvarname,
                                            datatype='i4',
                                            dimensions=dim,
                                            fill_value=0)
                var_cc.setncattr('long_name', long_name)
                var_cc.setncattr('units', '')
            grp[ncvarname][:] = flag['values']


def parse_cloudtype_from_frm4doas(main: np.ndarray, sub: np.ndarray,
                                  warn: np.ndarray) -> MDCloudResult:
    '''
    Load the cloud type results from a given FRM4DOAS file.

    @main main category flags as 2D array (n_sequences, 8)
    @sub sub category flags as 2D array (n_sequences, 8)
    @warn warning flags as 2D array (n_sequences, 8)
    '''
    result = MDCloudResult(main.shape[0])
    result.main['values'] = main
    result.sub['values'] = sub
    result.warn['values'] = warn
    return result


class FRM4DOASCloudResultPlotter():
    '''
    Class that nicely organizes all functions needed to create a cloud
    classification report for multiple FRM4DOAS input files.
    '''

    def __init__(self, frm4doas_fns: list, pdfstream: PdfPages):
        '''
        @frm4doas_fns
        @pdfstream
        '''
        self.pdfstream = pdfstream
        try:
            out_fn = pdfstream._filename
        except AttributeError:
            out_fn = "unknown"
        self.out_fn = out_fn

        # Placeholders
        self.load_frm4doas_data(frm4doas_fns)

    def load_frm4doas_data(self, frm4doas_fns: list):
        '''
        Load all necessary data from FRM4DOAS files.

        @frm4doas_fns
        '''
        data = load_multiple_frm4doas_data(frm4doas_fns, read_cloudflags=True)

        # Provide solar time to enable daily plots of measurements
        vec_solar_dt = np.vectorize(tc.calc_solar_dt)
        vec_epoch_to_dt = np.vectorize(tc.get_dt_from_unix_epoch)
        data['solar_dt'] = vec_solar_dt(data['dt'], data['lon'])
        # Get sequential mean time stamps for plotting using epoch format
        data['plot_dt'] = vec_epoch_to_dt(np.ma.mean(data['epoch'], axis=1))
        self.start_dt = data['solar_dt'][~data['dt'].mask][0]
        self.end_dt = data['solar_dt'][~data['dt'].mask][-1]

        # Mobile measurements?

        def all_equal(arr, tol=1e-8):
            return np.all(np.abs(arr - arr[0]) < tol)

        self.stationary = all_equal(data['lon']) and all_equal(data['lat'])
        self.data = data
        self.cloud_type = parse_cloudtype_from_frm4doas(data['main'],
                                                        data['sub'],
                                                        data['warn'])

    def create_header(self):
        '''
        Create header pages from the given data set.
        '''
        # Prepare header page
        header_list = []
        header_list.append(f"Institute: {self.data['institution'][0]}")
        header_list.append(f"Campaign: {self.data['campaign_name'][0]}")
        header_list.append(f"Station: {self.data['station_name'][0]}")
        header_list.append("")
        if self.stationary:
            header_list.append("Location of the Instrument:")
            header_list.append(f"Longitude: {self.data['lon'][0, 0]:.3f}")
            header_list.append(f"Latitude: {self.data['lat'][0, 0]:.3f}")
        header_list.append("")
        start = np.nanmin(self.data['dt'].compressed())
        end = np.nanmax(self.data['dt'].compressed())
        header_list.append(f"Data Start: {start:%Y-%m-%d %H:%M:%S}")
        header_list.append(f"Data End: {end:%Y-%m-%d %H:%M:%S}")
        header_list.append(f"Saved as: {self.out_fn}")
        header_list.append("")
        header_list.append("List of input files:")
        header_list.append(file_tools.filelist_to_str(self.data['filename'],
                                                      grouping=True))
        # Create an empty page
        plot_tools.add_text_page(self.pdfstream, text='\n'.join(header_list))

    def create_legend(self, flag_types: list=None, ms: int=7):
        '''
        Create legend pages (horizontal and vertical)

        @flag_types (default=None) can be any sublist of available cloud types
            being main, sub and warn (see MDCloudResult.keys)
        @ms (default=7) is the marker size used in the legend
        '''
        MAX_CHARACTER_PER_ROW = 60
        legend_handles = {}
        legend_titles = {}
        legend_ncols = {}
        if flag_types is None:
            flag_types = self.cloud_type.keys
        for flag_type in flag_types:
            cur_flag = getattr(self.cloud_type, flag_type)
            legend_handles[flag_type] = []
            label_chars = 0
            for flag_idx, flag_name in enumerate(cur_flag['labels']):
                if flag_name == 'empty':
                    continue
                color = cur_flag['colors'][flag_idx]
                marker = cur_flag['markers'][flag_idx]
                cur_line = Line2D([0], [0], marker=marker, ms=ms, mec=color,
                                  ls='', c=color, label=flag_name)
                legend_handles[flag_type].append(cur_line)
                label_chars += len(flag_name)
            legend_titles[flag_type] = cur_flag['long_name']
            avg_chars_per_label = label_chars / len(legend_handles[flag_type])
            ncols = np.round(MAX_CHARACTER_PER_ROW / avg_chars_per_label)
            legend_ncols[flag_type] = ncols
        n_types = len(flag_types)
        fig, axes = plt.subplots(figsize=(8.27, 11.69), ncols=n_types)
        fig.patch.set_alpha(0.0)
        if n_types == 1:  # axes should always be a list
            axes = [axes]
        for ax in axes:
            ax.axis('off')
            ax.set_position([0.01, 0, 1, 0.99])
        for idx, flag_type in enumerate(flag_types):
            legend_position = (0.0, 1.0 - idx / n_types)
            leg = axes[idx].legend(handles=legend_handles[flag_type],
                                   title=legend_titles[flag_type],
                                   loc='upper left', frameon=False,
                                   bbox_to_anchor=legend_position)
            leg._legend_box.align = "left"
        self.pdfstream.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(figsize=(8.27, 11.69), ncols=n_types)
        fig.patch.set_alpha(0.0)
        if n_types == 1:  # axes should always be a list
            axes = [axes]
        for ax in axes:
            ax.axis('off')
            ax.set_position([0.01, 0, 1, 0.99])
        for idx, flag_type in enumerate(flag_types):
            legend_position = (0.0, 1.0 - idx / n_types)
            leg = axes[idx].legend(handles=legend_handles[flag_type],
                                   title=legend_titles[flag_type],
                                   loc='upper left', frameon=False,
                                   bbox_to_anchor=legend_position,
                                   ncols=legend_ncols[flag_type])
            leg._legend_box.align = "left"
        self.pdfstream.savefig(fig)
        plt.close(fig)

    def create_overview_plot(self, flag_types: list=None, **kwargs):
        '''
        Cloud classification overview plot

        @flag_types (default=None) can be any sublist of available cloud types
            being main, sub and warn (see MDCloudResult.keys)
        @kwargs:
            custom_datelim: tuple with two values for having custom date limits
                            in the overview plot, default is time range in
                            self.data +/- 2 days
        '''
        if flag_types is None:
            flag_types = self.cloud_type.keys
        n_types = len(flag_types)

        current_day = self.start_dt.replace(hour=0, minute=0, second=0,
                                            microsecond=0)
        # Group by days - could be done so much easier with pandas, but
        # including pandas now just for this plot seems unneccessary
        dates = []
        dates_seqs = []
        while current_day < self.end_dt:
            next_day = current_day + timedelta(days=1)
            valid = ~self.data['solar_dt'].mask
            tmp_mask = np.zeros_like(valid, dtype=bool)
            tmp_mask[valid] = self.data['solar_dt'][valid] > current_day
            tmp_mask[valid] &= self.data['solar_dt'][valid] < next_day
            valid &= tmp_mask
            valid_seq = np.unique(np.where(valid)[0])
            dates.append(current_day.replace(hour=12))
            dates_seqs.append(valid_seq)
            current_day = next_day
        datelim = kwargs.get('custom_datelim', (dates[0] - timedelta(days=2),
                                                dates[-1] + timedelta(days=2)))

        fig, axes = plt.subplots(figsize=(10, 2*n_types), nrows=n_types,
                                 sharex=True)
        fig.patch.set_alpha(0.0)
        if n_types == 1:  # axes should always be a list
            axes = [axes]
        for type_idx, flag_type in enumerate(flag_types):
            cur_flag = getattr(self.cloud_type, flag_type)
            dates_flag_count = []
            dates_noflag_count = []
            for idx, _ in enumerate(dates):
                cur_flag_count = np.sum(cur_flag['values'][dates_seqs[idx]],
                                        axis=0)
                seq_flag_count = np.sum(cur_flag['values'][dates_seqs[idx]],
                                        axis=1)
                cur_noflag_count = np.where(seq_flag_count.mask)[0].size
                dates_flag_count.append(cur_flag_count)
                dates_noflag_count.append(cur_noflag_count)
            dates_flag_count = np.array(dates_flag_count, dtype=float)
            dates_noflag_count = np.array(dates_noflag_count)
            # Create relative count for main flag_type
            if flag_type == 'main':
                daily_sum = np.sum(dates_flag_count, axis=1)
                dates_flag_count /= daily_sum[:, np.newaxis]
                dates_flag_count *= 100
                axes[type_idx].set_ylabel(f'Rel. Frequency ({flag_type})')
                axes[type_idx].yaxis.set_major_formatter(PercentFormatter())
                axes[type_idx].set_ylim(0, 100)
            else:
                axes[type_idx].set_ylabel(f'Abs. Frequency ({flag_type})')
            bottom_values = np.zeros(len(dates))  # Start from zero
            # Also show number of sequences without warnings
            if flag_type == 'warn':
                axes[type_idx].bar(dates, dates_noflag_count,
                                   label='no warnings', edgecolor='k',
                                   color='white', hatch='///',
                                   linewidth=0.4, bottom=bottom_values)
                bottom_values += dates_noflag_count
            for flag_idx, flag_name in enumerate(cur_flag['labels']):
                cur_date_flag_count = dates_flag_count[:, flag_idx]
                axes[type_idx].bar(dates, cur_date_flag_count,
                                   label=flag_name, edgecolor='k',
                                   color=cur_flag['colors'][flag_idx],
                                   linewidth=0.4, bottom=bottom_values)
                bottom_values += cur_date_flag_count
            axes[type_idx].grid(ls=':', color='gray')
        axes[0].set_title('Cloud classification overview')
        axes[-1].set_xlabel('Date')
        axes[-1].set_xlim(datelim)
        axes[-1].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].tick_params("x", labelrotation=90)

        self.pdfstream.savefig(dpi=300, bbox_inches='tight')

    def create_daily_plots(self, flag_types: list=None, ms: int=5):
        '''
        Create daily plots of the cloud classification results.

        @flag_types (default=None) can be any sublist of available cloud types
            being main, sub and warn (see MDCloudResult.keys)
        @ms (default=5) is the marker size used in the legend
        '''
        current_day = self.start_dt.replace(hour=0, minute=0, second=0,
                                            microsecond=0)
        if flag_types is None:
            flag_types = self.cloud_type.keys
        data_plotted = False
        while current_day < self.end_dt:
            next_day = current_day + timedelta(days=1)
            valid = ~self.data['solar_dt'].mask
            tmp_mask = np.zeros_like(valid, dtype=bool)
            tmp_mask[valid] = self.data['solar_dt'][valid] > current_day
            tmp_mask[valid] &= self.data['solar_dt'][valid] < next_day
            valid &= tmp_mask
            valid_seq = np.unique(np.where(valid)[0])

            n_types = len(flag_types)
            fig, axes = plt.subplots(figsize=(10, 2*n_types), nrows=n_types,
                                     sharex=True)
            fig.patch.set_alpha(0.0)
            if n_types == 1:  # axes should always be a list
                axes = [axes]
            for type_idx, flag_type in enumerate(flag_types):
                axes[type_idx].scatter([current_day], [0], color='#FFFFFF',
                                       marker=',')
                cur_flag = getattr(self.cloud_type, flag_type)
                ylabels = []
                nflags = np.where(np.array(cur_flag['labels']) != 'empty')
                nflags = nflags[0].size
                for flag_idx, flag_name in enumerate(cur_flag['labels']):
                    if flag_name == 'empty':
                        continue
                    ylabels.append(flag_name)
                    flag_filter = cur_flag['values'][valid_seq, flag_idx] == 1
                    if not np.any(flag_filter):
                        continue
                    plot_dt = self.data['plot_dt'][valid_seq][flag_filter]
                    plot_flagval = np.zeros_like(plot_dt, dtype=float)
                    plot_flagval += (nflags - flag_idx - 1)
                    axes[type_idx].plot(plot_dt, plot_flagval,
                                        marker=cur_flag['markers'][flag_idx],
                                        c=cur_flag['colors'][flag_idx],
                                        ms=ms, ls='')
                    data_plotted = True
                axes[type_idx].grid(ls=':', color='gray')
                axes[type_idx].set_ylim(0 - 0.5, nflags - 0.5)
                axes[type_idx].set_yticks(np.arange(nflags))
                wrapped_ylabels = []
                for ylabel in ylabels:
                    wrapped_ylabel = textwrap.wrap(ylabel[:27], width=14)
                    wrapped_ylabels.append('\n'.join(wrapped_ylabel))
                axes[type_idx].set_yticklabels(wrapped_ylabels[::-1])
                axes[type_idx].grid(ls=':', color='gray')
            axes[0].set_title(f"{current_day:%Y-%m-%d}")
            xtime_major = mdates.HourLocator(interval=6)
            xtime_minor = mdates.HourLocator(interval=1)
            xtime_format = mdates.DateFormatter('%Hh')
            axes[-1].xaxis.set_major_locator(xtime_major)
            axes[-1].xaxis.set_major_formatter(xtime_format)
            axes[-1].xaxis.set_minor_locator(xtime_minor)
            axes[-1].set_xlim(current_day, next_day)
            if data_plotted:
                self.pdfstream.savefig(fig)
            plt.close(fig)
            current_day = next_day


def run_md_cloud_classification(classification_cfg_fn: str,
                                threshold_cfg_fn: str,
                                data_dir: str,
                                run_calibration: bool=False,
                                calibration_report_fn: str=None,
                                classification_report_fn: str=None,
                                o4_vcd: float=1.3e43):
    with open(classification_cfg_fn, 'r', encoding="utf-8") as cfg:
        config = yaml.load(cfg, Loader=yaml.SafeLoader)
    with open(threshold_cfg_fn, 'r', encoding="utf-8") as cfg:
        thresholds_config = yaml.load(cfg, Loader=yaml.SafeLoader)

    data_fns = file_tools.get_filelist(data_dir, recursive=True,
                                       must_contain=['L15', '.nc'])
    if run_calibration:
        data = load_multiple_frm4doas_data(data_fns)
        if 'o4_vcd' in data:
            data['o4_damf'] = data['o4_dscd'] / data['o4_vcd']
        else:
            data['o4_damf'] = data['o4_dscd'] / o4_vcd
        cc = MAXDOASCloudClassification(config, thresholds_config)
        with PdfPages(calibration_report_fn) as pdfout:
            # Full controll which steps should be performed:
            cc.gen_thresholds(data['sza'], data['elev'])
            cc.set_classification_mask(data['sza'], data['elev'])
            cc.normalize_ci(data['sza'], data['elev'], data['ci'],
                            plot_stream=pdfout, verbose=True)
            cloud_type = cc.classify_ci_cloud(data['elev'], data['ci'],
                                              data['dt'], data['lon'])
            cc.normalize_o4(data['sza'], data['elev'], data['o4_damf'],
                            cloud_type, plot_stream=pdfout, verbose=True)
            cloud_type = cc.classify_o4_cloud(data['elev'], data['ci'],
                                              data['o4_damf'], cloud_type)
            cloud_type = cc.get_warning_flags(data['elev'], data['dt'],
                                              cloud_type)
        config['normalization_ci'] = cc.normalize_ci
        config['normalization_o4'] = cc.normalize_o4
        print('####################################\n'
              '##     Total cloud statistics     ##\n'
              '####################################')
        print(cloud_type)
        return
    for data_fn in data_fns:
        data = load_frm4doas_data(data_fn)
        if 'o4_vcd' in data:
            data['o4_damf'] = data['o4_dscd'] / data['o4_vcd']
        else:
            data['o4_damf'] = data['o4_dscd'] / o4_vcd
        cc = MAXDOASCloudClassification(config, thresholds_config)
        cloud_type = cc.classify_all(data['sza'], data['elev'], data['ci'],
                                     data['o4_damf'], data['dt'],
                                     data['lon'])
        save_cloudtype_to_frm4doas(data_fn, cloud_type)
    with PdfPages(classification_report_fn) as pdfout:
        plotter = FRM4DOASCloudResultPlotter(data_fns, pdfout)
        plotter.create_header()
        plotter.create_legend()
        plotter.create_overview_plot()
        plotter.create_daily_plots()
