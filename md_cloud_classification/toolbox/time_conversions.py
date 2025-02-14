'''
Created on 09.01.2025

@author: steffen.ziegler
'''
from datetime import datetime, timedelta
import numpy as np


def calc_solar_dt(dt, lon):
    '''
    Derive the solar datetime based on given longitude

    @dt datetime of a measurement
    @inst_lon longitude of the instrument location, needed for calculating the
        solar day and finding daily indices

    RETURNS solar_dt
    '''
    if dt is None or lon is None:
        return None
    # Fix longitude range to [-180, 180]
    if lon > 180:
        lon -= 360
    if lon < -180:
        lon += 360
    solar_tz_offset = lon * 12 / 180
    return dt + timedelta(hours=float(solar_tz_offset))


def check_dt_on_day(dt, date):
    '''
    Check if a given datetime object is on a given date.
    '''
    if dt is None:
        return False
    return date <= dt <= date + timedelta(days=1)


def get_unix_epoch(dt):
    '''
    Get UNIX EPOCH from a given datetime. UNIX EPOCH is the total amound of
    seconds since 1970-01-01 00:00:00
    '''
    if dt is None:
        return np.nan
    return (dt - datetime(1970, 1, 1)).total_seconds()
