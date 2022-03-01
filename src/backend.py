#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os
import sys
from re import split
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
p3_df = [pd.read_pickle(i) for i in ls_p3]
lear_df = [pd.read_pickle(i) for i in ls_lear]
dt_day = [{'label': f'{i: %Y-%m-%d}', 'value': f'{i}'} for i in p3_df[0].local_time.dt.floor('D').unique()]
dt_sensor = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in p3_df]
dt_aircraft = [{"label": 'P3B', 'value': 'P3B'}, {"label": 'Learjet', 'value': 'Learjet'}]


def get_sensors(aircraft: str) -> list:
    """

    :rtype: object
    """
    if aircraft == 'P3B':
        sensor_opt = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in p3_df]
        _day = sorted(set(np.concatenate([i.local_time.dt.floor('D').unique() for i in p3_df]).flat))
        _hour = sorted(set(np.concatenate([i.local_time.dt.floor('h').unique() for i in p3_df]).flat))
        date_opt = [{'label': f'{i: %Y-%m-%d}', 'value': i} for i in _day]
        # hour_opt = [{'label': f'{i: %H:%M}', 'value': i} for i in _hour]
        return sensor_opt, date_opt, #  hour_opt
    elif aircraft == 'Learjet':
        sensor_opt = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in lear_df
                      if i.attrs['type'] != 'Page0']
        _date = sorted(set(np.concatenate([i.local_time.dt.floor('D').unique() for i in lear_df]).flat))
        date_opt = [{'label': f'{i: %Y-%m-%d}', 'value': i} for i in _date]
        return sensor_opt, date_opt


def get_hour(aircraft, ls_sensor, day):
    if aircraft == 'P3B':
        ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                                           for i in p3_df if i.attrs['type'] in ls_sensor]
        _hour = sorted(set(np.concatenate([i['local_time'].dt.floor('h').unique() for i in ls_df]).flat))
        hour_opt = [{'label': f'{i: %H:%M}', 'value': i} for i in _hour]
        return hour_opt
    if aircraft == 'Learjet':
        ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                                           for i in lear_df if i.attrs['type'] in ls_sensor]
        _hour = sorted(set(np.concatenate([i['local_time'].dt.floor('h').unique() for i in ls_df]).flat))
        hour_opt = [{'label': f'{i: %H:%M}', 'value': i} for i in _hour]
        return hour_opt


def get_minutes(aircraft, ls_sensor, day, _hour):
    if aircraft == 'P3B':
        ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                 for i in p3_df if i.attrs['type'] in ls_sensor]
        ls_df = [i[i['local_time'].dt.hour == pd.to_datetime(_hour).hour]
                 for i in ls_df]
        _min = sorted(set(np.concatenate([i['local_time'].dt.floor('min').unique() for i in ls_df]).flat))
        min_opt = [{'label': f'{i: %M}', 'value': i} for i in _min]
        return min_opt
    elif aircraft == 'Learjet':
        if aircraft == 'Learjet':
            ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                     for i in lear_df if i.attrs['type'] in ls_sensor]
            ls_df = [i[i['local_time'].dt.hour == pd.to_datetime(_hour).hour]
                     for i in ls_df]
            _min = sorted(set(np.concatenate([i['local_time'].dt.floor('min').unique() for i in ls_df]).flat))
            min_opt = [{'label': f'{i: %M}', 'value': i} for i in _min]
            return min_opt


def get_seconds(aircraft, ls_sensor, day, _hour, minute):
    if aircraft == 'P3B':
        ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                 for i in p3_df if i.attrs['type'] in ls_sensor]
        ls_df = [i[i['local_time'].dt.hour == pd.to_datetime(_hour).hour]
                 for i in ls_df]
        ls_df = [i[i['local_time'].dt.minute == pd.to_datetime(minute).minute]
                 for i in ls_df]
        _secs = sorted(set(np.concatenate([i['local_time'].dt.floor('s').unique() for i in ls_df]).flat))
        return min(_secs).second, max(_secs).second
    elif aircraft == 'Learjet':
        ls_df = [i[i['local_time'].dt.date == pd.to_datetime(day)]
                 for i in lear_df if i.attrs['type'] in ls_sensor]
        ls_df = [i[i['local_time'].dt.hour == pd.to_datetime(_hour).hour]
                 for i in ls_df]
        ls_df = [i[i['local_time'].dt.minute == pd.to_datetime(minute).minute]
                 for i in ls_df]
        _secs = sorted(set(np.concatenate([i['local_time'].dt.floor('s').unique() for i in ls_df]).flat))
        return min(_secs).second, max(_secs).second


def main():
    pass


if __name__ == '__main__':
    main()