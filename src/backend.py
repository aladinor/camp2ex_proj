#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os
import sys
import plotly.graph_objects as go
from re import split

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
ls_p3_merged = glob.glob(f'{path_data}/data/01_SECOND.P3B_MRG/MERGE/all/p3b*.pkl')
p3_df = [pd.read_pickle(i) for i in ls_p3]
lear_df = [pd.read_pickle(i) for i in ls_lear]
dt_day = [{'label': f'{i: %Y-%m-%d}', 'value': f'{i}'} for i in p3_df[0].local_time.dt.floor('D').unique()]
dt_sensor = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in p3_df]
dt_aircraft = [{"label": 'P3B', 'value': 'P3B'}, {"label": 'Learjet', 'value': 'Learjet'}]


def psd_fig(_idx, ls_df, aircraft):
    layout = go.Layout(autosize=True, width=550, height=550,
                       title={'text': f"{_idx: %Y-%m-%d %H:%M:%S} - {aircraft} - Local time", 'x': 0.5,
                              'xanchor': 'center'})
    fig = go.Figure(layout=layout)
    for i in ls_df:
        x = i.attrs['sizes']
        y = i[i.index == _idx.tz_convert('UTC')].filter(like='nsd').iloc[0].values
        y = np.where(y > 0, y, np.nan)
        ax1 = fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=i.attrs['type']))
    ax1.update_traces(mode="lines", line_shape="vh")
    ax1.update_yaxes(title_text="Concentration (#L-1 um-1)", type="log", showgrid=False, exponentformat='power',
                     showexponent='all')
    ax1.update_xaxes(title_text="Diameter (um)", type="log", exponentformat='power', showexponent='all')
    ax1.update_layout(legend=dict(y=0.99, x=0.65))
    return fig


def plot_map(aircraft, date):
    if not date:
        date = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')

    if not aircraft or aircraft == "Learjet":
        df = lear_df[-1]
    else:
        df = pd.read_pickle(ls_p3_merged[0])

    df = df.loc[df['local_time'].dt.date == date, df.columns]
    df.loc[df['Lat'] == 0, 'Lat'] = np.nan
    df.loc[df['Long'] == 0, 'Long'] = np.nan
    lon_cent = df['Long'].mean()
    lat_cent = df['Lat'].mean()
    plane_lat = df.loc[df['local_time'] == date, 'Lat']
    plane_lon = df.loc[df['local_time'] == date, 'Long']
    layout = go.Layout(autosize=True, width=550, height=550)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scattermapbox(mode='lines',
                                   lon=df['Long'],
                                   lat=df['Lat'],
                                   showlegend=False
                                   )
                  )

    fig.add_trace(go.Scattermapbox(
        lat=plane_lat.values,
        lon=plane_lon.values,
        marker=dict(size=10, color='black', opacity=1),
        showlegend=False
    ))
    fig.update_layout(mapbox_style="open-street-map", mapbox_center_lat=lat_cent, mapbox_center_lon=lon_cent,
                      mapbox=dict(zoom=5), )

    return fig


def get_sensors(aircraft: str) -> tuple[list[dict[str:str]], list[dict[str:str]]]:
    """

    :rtype: object
    """
    if aircraft == 'P3B':
        sensor_opt = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in p3_df]
        _day = sorted(set(np.concatenate([i.local_time.dt.floor('D').unique() for i in p3_df]).flat))
        _hour = sorted(set(np.concatenate([i.local_time.dt.floor('h').unique() for i in p3_df]).flat))
        date_opt = [{'label': f'{i: %Y-%m-%d}', 'value': i} for i in _day]
        return sensor_opt, date_opt
    elif aircraft == 'Learjet':
        sensor_opt = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in lear_df
                      if i.attrs['type'] != 'Page0']
        _date = sorted(set(np.concatenate([i.local_time.dt.floor('D').unique() for i in lear_df]).flat))
        date_opt = [{'label': f'{i: %Y-%m-%d}', 'value': i} for i in _date]
        return sensor_opt, date_opt


def get_hour(aircraft, ls_sensor, day) -> list[dict[str:str]]:
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


def plot_nsd(aircraft, ls_sensor, _hour, minute, second):
    if (aircraft is None) or (_hour is None) or (minute is None) or (second is None):
        return psd_fig(_idx=pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila'),
                       ls_df=lear_df, aircraft='Learjet')

    if aircraft == 'P3B':
        ls_df = [i for i in p3_df if i.attrs['type'] in ls_sensor]
        _idx = pd.to_datetime(minute) + pd.to_timedelta(int(second), unit='s')
    else:
        ls_df = [i for i in lear_df if i.attrs['type'] in ls_sensor]
        _idx = pd.to_datetime(minute) + pd.to_timedelta(int(second), unit='s')

    return psd_fig(_idx=_idx, ls_df=ls_df, aircraft=aircraft)


def main():
    pass


if __name__ == '__main__':
    main()
