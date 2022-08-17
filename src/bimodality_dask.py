#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sqlalchemy.exc import OperationalError
# import xarray as xr
import itertools
from pytmatrix import tmatrix_aux, refractive, tmatrix, radar
from pymiecoated import Mie
from scipy.constants import c
from dask import delayed, compute
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from re import split
import matplotlib.dates as mdates

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']

cdict = {'red': ((0., 1, 1),
                 (0.05, 1, 1),
                 (0.11, 0, 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (0.05, 1, 1),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (0.05, 1, 1),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)


def get_data(instrument='Lear', temp=2):
    """

    :param instrument: aircraft
    :param temp: temperature for filtering
    :return: list of available dataframe in CAMP2ex
    """
    if instrument == 'Lear':
        ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
        ls_lear = [i for i in ls_lear if not i.split('/')[-1].startswith('Page0')]
        ls_temp = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/Page0*.pkl')[0]
        ls_lear.append(ls_temp)
        lear_df = [pd.read_pickle(i) for i in ls_lear]
        _attrs = [i.attrs for i in lear_df[:-1]]
        if temp:
            lear_df = [pd.merge(i, lear_df[-1]['Temp'], right_index=True, left_index=True) for i in lear_df[:-1]]
            lear_df = [i[i['Temp'] > temp] for i in lear_df]
        for i, attrs in enumerate(_attrs):
            lear_df[i].attrs = attrs
        lear_dd = [dd.from_pandas(i, npartitions=1) for i in lear_df]
        del lear_df
        return lear_dd
    elif instrument == 'P3B':
        ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
        p3_merged = glob.glob(f'{path_data}/data/01_SECOND.P3B_MRG/MERGE/all/*pkl')
        p3_temp = pd.read_pickle(p3_merged[0])
        p3_df = [pd.read_pickle(i) for i in ls_p3]
        _attrs = [i.attrs for i in p3_df]
        p3_df = [pd.merge(i, p3_temp[' Static_Air_Temp_YANG_MetNav'], left_index=True, right_index=True) for i in p3_df]
        temp = 2
        for i, df in enumerate(p3_df):
            df.attrs = _attrs[i]
            df.rename(columns={' Static_Air_Temp_YANG_MetNav': 'Temp'}, inplace=True)
            if temp:
                df = df[df['Temp'] >= temp]
            p3_df[i] = df
        p3_dd = [dd.from_pandas(i, npartitions=1) for i in p3_df]
        del p3_df
        return p3_dd
    else:
        raise TypeError(f"{instrument} not available. Use Lear or P3B")


def change_cols(df):
    bin_cent = df.attrs['bin_cent']
    cols = df.columns
    new_cols = {cols[i]: bin_cent[i] for i in range(len(cols))}
    df = df.rename(columns=new_cols)
    return df


@delayed
def change_cols(df):
    bin_cent = df.attrs['bin_cent']
    cols = df.columns
    new_cols = {cols[i]: bin_cent[i] for i in range(len(cols))}
    df = df.rename(columns=new_cols)
    return df


def filt_by_instrument(ls_df, hawk=False):
    if not hawk:
        ls_df = [i for i in ls_df if not i.attrs['instrument'].startswith('Hawk')]
        return ls_df
    else:
        return ls_df


def filt_by_cols(ls_df):
    # if cols:
    cols = [[j for j in i.columns if j.startswith('nsd')] for i in ls_df]
    ls_df = [i[cols[j]] for j, i in enumerate(ls_df)]
    ls_df = [change_cols(i) for i in ls_df]
    return ls_df


def get_intervals(ls_df):
    cols = [[pd.Interval(j.attrs['sizes'][i], j.attrs['sizes'][i + 1]) for i in
             range(len(j.attrs['sizes']) - 1)] for j in ls_df]
    cols = [j + [pd.Interval(ls_df[i].attrs['sizes'][-1], list(ls_df[i].attrs['dsizes'].keys())[-1])]
            for i, j in enumerate(cols)]
    return cols


def apply_wgt(df, ovr_upp, ovr_lower):
    if (df['2DS10'] != 0) & (df['HVPS'] != 0):
        df['2ds10_wgt'] = df['2DS10'] * df['2DS10'].index.values / (ovr_upp - ovr_lower)
        df['hvps_wgt'] = df['HVPS'] * (df['HVPS'].index.values - ovr_lower) / (ovr_upp - ovr_lower)
        df['nd_res'] = df[['2ds10_wgt', 'hvps_wgt']].dropna(how='all').sum(1)
        return df['nd_res']
    elif (df['2DS10'] == 0) & (df['HVPS'] != 0):
        return df['HVPS']
    elif (df['2DS10'] != 0) & (df['HVPS'] == 0):
        return df['HVPS']
    else:
        return np.nan


def linear_wgt(df1, df2, ovr_upp=1200, ovr_lower=800, method='linear'):
    """

    :param method: method to apply. linear applies Leroy et al. 2014. swal Applies Snesbitt method
    :param df1: pandas series with the small diameter sizes (e.g. 2DS10)
    :param df2: pandas series with the small diameter sizes (e.g. HVPS)
    :param ovr_upp: upper limit for overlapping
    :param ovr_lower: lower limit for overlapping
    :return: pd series with a composite PSD
    """
    df1.index = df1.attrs['2DS10']['intervals']
    df2.index = df2.attrs['HVPS']['intervals']
    cond1 = (df1.index.mid >= ovr_lower) & (df1.index.mid <= ovr_upp)
    cond2 = (df2.index.mid >= ovr_lower) & (df2.index.mid <= ovr_upp)
    _nd_uppr = df1[cond1]
    _nd_lower = df2[cond2]

    nd_overlap = pd.concat([_nd_uppr.reindex(np.arange(ovr_lower, ovr_upp, 5)),
                            _nd_lower.reindex(np.arange(ovr_lower, ovr_upp, 5))], axis=1)
    if method == 'linear':
        nd_overlap['2ds10_wgt'] = nd_overlap['2DS10'] * (ovr_upp - nd_overlap['2DS10'].index.values) / \
                                  (ovr_upp - ovr_lower)
        nd_overlap['hvps_wgt'] = nd_overlap['HVPS'] * (nd_overlap['HVPS'].index.values - ovr_lower) / \
                                 (ovr_upp - ovr_lower)
        nd_overlap['nd_res'] = nd_overlap[['2ds10_wgt', 'hvps_wgt']].dropna(how='all').sum(1)
        nd_overlap = nd_overlap.reindex(df1[cond1].index.mid)
        nd_overlap.index = df1[cond1].index
        res = pd.concat([df1[df1.index.mid <= ovr_lower], nd_overlap['nd_res'], df2[df2.index.mid >= ovr_upp]])
        dd = {i.mid: i.length for i in res.index}
        res.index = res.index.mid
        res.attrs['dsizes'] = dd
        return res

    elif 'snal':
        _sdd = nd_overlap.where(
            (nd_overlap['2DS10'] != 0) & (nd_overlap['HVPS'] != 0) & (nd_overlap['2DS10'].notnull()) &
            (nd_overlap['HVPS'] != 0).notnull()).dropna(how='any')
        _sdd['2ds10_wgt'] = _sdd['2DS10'] * (ovr_upp - _sdd['2DS10'].index.values) / (ovr_upp - ovr_lower)
        _sdd['hvps_wgt'] = _sdd['HVPS'] * (_sdd['HVPS'].index.values - ovr_lower) / (ovr_upp - ovr_lower)
        _sdd['nd_res'] = _sdd[['2ds10_wgt', 'hvps_wgt']].sum(1)

        _sdc = nd_overlap.where(((nd_overlap['2DS10'] == 0) | (nd_overlap['2DS10'].isnull())) &
                                (nd_overlap['HVPS'] != 0)).dropna(how='all')['HVPS']
        _sds = nd_overlap.where((nd_overlap['2DS10'] != 0) & ((nd_overlap['HVPS'] == 0) |
                                                              (nd_overlap['HVPS'].isnull()))).dropna(how='all')['2DS10']
        res = pd.concat([_sdd['nd_res'], _sdc, _sds]).sort_index()
        res = res.reindex(df1[cond1].index.mid)
        res.index = df1[cond1].index
        res = pd.concat([df1[df1.index.mid <= ovr_lower], res, df2[df2.index.mid >= ovr_upp]])
        dd = {i.mid: i.length for i in res.index}
        res.index = res.index.mid
        res.attrs['dsizes'] = dd
        return res


def vel(d):
    return -0.1021 + 4.932 * d - 0.955 * d ** 2 + 0.07934 * d ** 3 - 0.0023626 * d ** 4


def pds_parameters(nd):
    """
    Compute the psd parameters
    :param nd: partice size distribution in # L-1 um-1
    :return: list with lwc, dm, nw, z, and r
    """
    try:
        d = np.fromiter(nd.index, dtype=float) / 1e3  # diameter in millimeters
        dd = np.fromiter(nd.attrs['dsizes'].values(), dtype=float)  # d_size in um
        lwc = (np.pi / (6 * 1000)) * np.sum((nd * 1e6) * d ** 3 * (dd * 1e-3))  # g / m3
        dm = np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd)  # mm
        nw = 1e3 * (4 ** 4 / np.pi) * (lwc / dm ** 4)
        z = np.sum(nd * d ** 6 * dd)
        r = np.pi * 6e-4 * np.sum(nd * d ** 3 * vel(d) * dd)
        return [lwc, dm, nw, z, r]
    except ZeroDivisionError:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]


def bcksct(ds, instrument, _lower, _upper, ar=1, j=0) -> pd.DataFrame:
    """

    :param _upper: upper diameter of the pds
    :param _lower: upper diameter of the pds
    :param instrument:
    :param ds: numpy array of particle diameters. should be in millimeters
    :param ar: axis ratio of the particle
    :param j: Zenith angle input
    :return:
    """
    x_ku = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_Ku
    x_ka = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_Ka
    x_w = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_W
    # Tmatrix calculations
    tmat_ku = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_Ku,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ku], axis_ratio=1.0 / ar, thet0=j,
                                                   thet=180 - j,
                                                   phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
               i in ds]
    tmat_ka = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_Ka,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ka], axis_ratio=1.0 / ar, thet0=j,
                                                   thet=180 - j,
                                                   phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
               i in ds]
    tmat_w = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_W,
                                                  m=refractive.m_w_0C[tmatrix_aux.wl_W], axis_ratio=1.0 / ar, thet0=j,
                                                  thet=180 - j,
                                                  phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
              i in ds]

    # Mie calculations
    mie_ku = [Mie(x=x_ku[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ku]).qb() * np.pi * (i / 2.) ** 2 for w, i in
              enumerate(ds)]
    mie_ka = [Mie(x=x_ka[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ka]).qb() * np.pi * (i / 2.) ** 2 for w, i in
              enumerate(ds)]
    mie_w = [Mie(x=x_w[w], m=refractive.m_w_0C[tmatrix_aux.wl_W]).qb() * np.pi * (i / 2.) ** 2 for w, i in
             enumerate(ds)]
    df_scatter = pd.DataFrame(
        {'T_mat_Ku': tmat_ku, 'T_mat_Ka': tmat_ka, 'T_mat_W': tmat_w, 'Mie_Ku': mie_ku, 'Mie_Ka': mie_ka,
         'Mie_W': mie_w}, index=ds)
    path_db = f'{path_data}/db'
    str_db = f"sqlite:///{path_db}/backscatter{_lower}_{_upper}.sqlite"
    df_scatter.to_sql(f'{instrument}', con=str_db, if_exists='replace')
    return df_scatter


def ref_calc(nd, _lower, _upper, mie=False):
    ds = np.fromiter(nd.attrs['dsizes'].keys(), dtype=float) / 1e3
    try:
        path_db = f'{path_data}/db'
        make_dir(path_db)
        str_db = f"sqlite:///{path_db}/backscatter{_lower}_{_upper}.sqlite"
        backscatter = pd.read_sql(f"{nd.attrs['instrument']}", con=str_db)
    except OperationalError:
        backscatter = bcksct(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper)
    dsizes = np.fromiter(nd.attrs['dsizes'].values(), dtype=float)
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_Ku'] * nd.values * 1000 * dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_Ka'] * nd.values * 1000 * dsizes)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_W'] * nd.values * 1000 * dsizes)
        return pd.Series({'Ku': 10 * np.log10(z_ku), 'Ka': 10 * np.log10(z_ka), 'W': 10 * np.log10(z_w)},
                         name=nd.attrs['instrument'])
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * backscatter['T_mat_Ku'] * nd.values * 1000 * dsizes
        z_ku.index = ds
        z_ku.name = 'z_Ku'
        # zku_r = np.sum(nd.values * 1000 * (ds ** 6) * dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * backscatter['T_mat_Ka'] * nd.values * 1000 * dsizes
        z_ka.index = ds
        z_ka.name = 'z_Ka'
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * backscatter['T_mat_W'] * nd.values * 1000 * dsizes
        z_w.index = ds
        z_w.name = 'z_Ka'
        instr = ['z_Ku', 'z_Ka', 'z_W']
        df_z = pd.concat([z_ku, z_ka, z_w], axis=0, keys=instr, levels=[instr])
        return df_z


def compute_hr(t, td):
    """
    Computes relative humidity using magnus approximation using
    https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    :param t: temperature
    :param td: dew point temperature
    :return: relative humidity
    """
    b = 17.625
    _c = 243.04
    return 100 * np.exp((_c * b * (td - t) / ((_c + t) * (_c + td))))


def get_add_data(aircraft: 'str', indexx) -> pd.DataFrame:
    path_db = f'{path_data}/db'
    str_db = f"sqlite:///{path_db}/camp2ex.sqlite"
    if aircraft == 'Lear':
        df_add = pd.read_sql_query("SELECT time, Temp, Dew, Palt, NevLWC, VaV  FROM Page0_Learjet", con=str_db)
        df_add['time'] = df_add['time'].apply(pd.Timestamp).apply(lambda x: x.tz_localize('UTC'))
        df_add['RH'] = df_add[['Temp', 'Dew']].apply(lambda x: compute_hr(t=x['Temp'], td=x['Dew']), axis=1)
        df_add['RH'] = df_add['RH'].where(df_add['RH'] <= 100, 100)
        df_add.index = df_add['time']
        df_add.drop('time', axis=1, inplace=True)
        cols = ['temp', 'dew_point', 'altitude(ft)', 'Nev_lwc', 'vertical_vel', 'RH']
        new_cols = {j: cols[i] for i, j in enumerate(list(df_add.columns))}
        df_add.rename(columns=new_cols, inplace=True)
        df_add = df_add[(df_add.index >= indexx.min().strftime('%Y-%m-%d %X')) &
                        (df_add.index <= indexx.max().strftime('%Y-%m-%d %X'))]
        return df_add
    else:
        df_add = pd.read_sql_query("SELECT pbm.'time', pbm.' Total_Air_Temp_YANG_MetNav', pbm.' "
                                   "Dew_Point_YANG_MetNav', pbm.' GPS_Altitude_YANG_MetNav', pbm.' LWC_gm3_LAWSON',  "
                                   "pbm.' Vertical_Speed_YANG_MetNav', pbm.' Relative_Humidity_YANG_MetNav' FROM "
                                   "p3b_merge pbm", con=str_db)
        df_add['time'] = df_add['time'].apply(pd.Timestamp).apply(lambda x: x.tz_localize('UTC'))
        df_add.index = df_add['time']
        df_add.drop('time', axis=1, inplace=True)
        cols = ['temp', 'dew_point', 'altitude(m)', 'Lawson_lwc', 'vertical_vel', 'RH']
        new_cols = {j: cols[i] for i, j in enumerate(list(df_add.columns))}
        df_add.rename(columns=new_cols, inplace=True)
        return df_add


def main():
    aircraft = 'P3B'
    _upper = 800
    _lower = 400
    ls_df = get_data(aircraft, temp=2)
    ls_df = filt_by_instrument(ls_df)
    ls_df = filt_by_cols(ls_df)
    ls_df = compute(*ls_df)
    intervals = get_intervals(ls_df)

    for i, inter in enumerate(intervals):
        ls_df[i].attrs['intervals'] = inter
    instr = [i.attrs['instrument'] for i in ls_df]
    attrs = [i.attrs for i in ls_df]
    dt_attrs = {instr[i]: j for i, j in enumerate(attrs)}
    df_concat = pd.concat(compute(*ls_df), axis=1, keys=instr, levels=[instr])
    df_concat.attrs = dt_attrs

    # rdm_idx = pd.date_range(start='2019-09-07 2:31:45', periods=150, tz='UTC', freq='S')  # for Lear
    # rdm_idx = pd.date_range(start='2019-09-09 0:51:57', periods=10, tz='UTC', freq='S')  # for Lear
    rdm_idx = pd.date_range(start='2019-09-06 23:58:30', periods=60, tz='UTC', freq='S')  # for P3B
    # rdm_idx = df_concat.index
    indexx = rdm_idx

    ls = []
    ls_z = []
    params = pd.DataFrame(index=indexx, columns=['lwc', 'dm', 'nw', 'z', 'r'])
    params2 = pd.DataFrame(index=indexx, columns=['lwc', 'dm', 'nw', 'z', 'r'])
    for i in indexx:
        df = df_concat.loc[i]
        res1 = df.unstack().T
        if res1['2DS10'].dropna(how='any').empty:
            df1 = pd.Series(index=df_concat.attrs['2DS10']['bin_cent'], dtype='float64', name='2DS10')
        elif len(res1['2DS10'].dropna(how='any')) < len(df_concat.attrs['2DS10']['bin_cent']):
            df1 = pd.Series(index=df_concat.attrs['2DS10']['bin_cent'], dtype='float64', name='2DS10')
        else:
            df1 = res1['2DS10'].dropna(how='any')

        if res1['HVPS'].dropna(how='any').empty:
            df2 = pd.Series(index=df_concat.attrs['HVPS']['bin_cent'], dtype='float64', name='HVPS')
        elif len(res1['HVPS'].dropna(how='any')) < len(df_concat.attrs['HVPS']['bin_cent']):
            df2 = pd.Series(index=df_concat.attrs['HVPS']['bin_cent'], dtype='float64', name='HVPS')
        else:
            df2 = res1['HVPS'].dropna(how='any')
        df1.attrs = df_concat.attrs
        df2.attrs = df_concat.attrs

        comp_pds = linear_wgt(df1=df1, df2=df2, ovr_upp=_upper, ovr_lower=_lower, method='snal')
        comp_pds.attrs['instrument'] = 'Composite_PSD'
        params.loc[i] = pds_parameters(comp_pds)
        ls_z.append(ref_calc(comp_pds, _lower=_lower, _upper=_upper))
        ls.append(comp_pds)
    attrs_merged = comp_pds.attrs['dsizes']
    df_reflectivity = pd.concat(ls_z, axis=1).T.set_index(indexx)
    df_merged = pd.concat(ls, axis=1).T.set_index(indexx)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(params.index, params['lwc'], label='800-1200')
    ax.plot(params2.index, params2['lwc'], label='400-1200')
    ax.legend()
    # x = np.linspace(*ax.get_xlim())
    # ax.plot(x, x)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.show()
    print(1)

    df_add = get_add_data(aircraft, indexx=indexx)

    df_merged.attrs['dsizes'] = attrs_merged
    df_day = df_merged.groupby(df_merged.index.floor('d'))
    keys = list(df_day.groups.keys())
    del df_day

    # for key in keys:
    #     airc = ls_df[0].attrs['aircraft']
    #     df = df_merged.groupby(df_merged.index.floor('d')).get_group(key)
    #     df = df[df > 0]
    #     fig, ax = plt.subplots(figsize=(14, 6))
    #     cbar = ax.pcolormesh(df.index, df.columns * 1e-3, np.log10(df.T * 1e6), vmin=0, vmax=10, cmap='jet')
    #     plt.colorbar(cbar, ax=ax, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
    #     ax.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
    #     ax.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
    #     ax.set_title('$N(D), \log_{10} (\# \ m^{-3} mm^{-1}) $', position=(0.8, 0.1), fontsize='x-large')
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    #     ax.set_yscale('log')
    #     title = f"{key: %Y-%m-%d} UTC - {aircraft}"
    #     fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    #     plt.tight_layout()
    #     # path_save = f'{path_data}/results/bimodality/flight/{aircraft}'
    #     # make_dir(path_save)
    #     # fig.savefig(f"{path_save}/{aircraft}_{key:%Y%m%d}.jpg")
    #     plt.show()
    #     print(1)

    for key in keys:
        airc = ls_df[0].attrs['aircraft']
        df = df_merged.groupby(df_merged.index.floor('d')).get_group(key)
        df_z = df_concat.groupby(df_concat.index.floor('d')).get_group(key)
        # df_z = df_z.loc[(df_z.index > '2019-09-07 02:33:00') & (df_z.index < '2019-09-07 02:33:55')]
        df_z = df_z.loc[(df_z.index > indexx.min()) & (df_z.index < indexx.max())]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        cbar = ax1.pcolormesh(df.index, df.columns * 1e-3, np.log10(df.T * 1e6), vmin=0, vmax=10, cmap=my_cmap)
        cbar2 = ax2.pcolormesh(df_z['2DS10'].index, df_z['2DS10'].columns * 1e-3, np.log10(df_z['2DS10'].T * 1e6),
                               vmin=0,
                               vmax=10, cmap=my_cmap)
        cbar3 = ax3.pcolormesh(df_z['HVPS'].index, df_z['HVPS'].columns * 1e-3, np.log10(df_z['HVPS'].T * 1e6), vmin=0,
                               vmax=10, cmap=my_cmap)

        plt.colorbar(cbar, ax=ax1, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        plt.colorbar(cbar2, ax=ax2, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        plt.colorbar(cbar3, ax=ax3, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        ax1.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
        ax1.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
        ax1.set_title('$N(D), \log_{10} (\# \ m^{-3} mm^{-1}) $', position=(0.8, 0.1), fontsize='x-large')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax2.set_title('2DS10')
        ax3.set_title('HVPS')
        ax1.set_xlim(df_merged.index.min(), df_merged.index.max())
        ax2.set_xlim(df_merged.index.min(), df_merged.index.max())
        ax3.set_xlim(df_merged.index.min(), df_merged.index.max())
        ax1.set_ylim(0.01, 40)
        ax2.set_ylim(0.01, 40)
        ax3.set_ylim(0.01, 40)

        title = f"{key: %Y-%m-%d} UTC - {aircraft}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        # path_save = f'{path_data}/results/bimodality/flight/{aircraft}'
        # make_dir(path_save)
        # fig.savefig(f"{path_save}/{aircraft}_{key:%Y%m%d}.jpg")
        plt.show()
        print(1)

    for key in keys:
        airc = ls_df[0].attrs['aircraft']
        df = df_merged.groupby(df_merged.index.floor('d')).get_group(key)
        df_z = df_reflectivity.groupby(df_merged.index.floor('d')).get_group(key)
        df = df[df > 0]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        cbar = ax1.pcolormesh(df.index, df.columns * 1e-3, np.log10(df.T * 1e6), vmin=0, vmax=10, cmap=my_cmap)
        cbar2 = ax2.pcolormesh(df_z['z_Ku'].index, df_z['z_Ku'].columns, 10 * np.log10(df_z['z_Ku'].T), vmin=-10,
                               vmax=50, cmap='jet')
        cbar3 = ax3.pcolormesh(df_z['z_Ka'].index, df_z['z_Ka'].columns, 10 * np.log10(df_z['z_Ka'].T), vmin=-10,
                               vmax=50, cmap='jet')
        cbar4 = ax4.pcolormesh(df_z['z_W'].index, df_z['z_W'].columns, 10 * np.log10(df_z['z_W'].T), vmin=-10,
                               vmax=50, cmap='jet')

        plt.colorbar(cbar, ax=ax1, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        plt.colorbar(cbar2, ax=ax2, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        plt.colorbar(cbar3, ax=ax3, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        plt.colorbar(cbar4, ax=ax4, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        ax1.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
        ax1.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
        ax1.set_title('$N(D), \log_{10} (\# \ m^{-3} mm^{-1}) $', position=(0.8, 0.1), fontsize='x-large')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax2.set_title('Ku-band radar reflectivity')
        ax3.set_title('Ka-band radar reflectivity')
        ax4.set_title('W-band radar reflectivity')
        title = f"{key: %Y-%m-%d} UTC - {aircraft}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        # path_save = f'{path_data}/results/bimodality/flight/{aircraft}'
        # make_dir(path_save)
        # fig.savefig(f"{path_save}/{aircraft}_{key:%Y%m%d}.jpg")
        plt.show()
        print(1)


if __name__ == '__main__':
    main()
