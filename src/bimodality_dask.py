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
        return p3_df
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


def linear_wgt(df1, df2, ovr_upp=1200, ovr_lower=800):
    """

    :param df1: pandas series with the small diameter sizes (e.g. 2DS10)
    :param df2: pandas series with the small diameter sizes (e.g. HVPS)
    :param ovr_upp: upper limit for overlapping
    :param ovr_lower: lower limit for overlapping
    :return: pd series with a composite PSD
    """
    cond1 = (df1.index > ovr_lower) & (df1.index < ovr_upp)
    cond2 = (df2.index > ovr_lower) & (df2.index < ovr_upp)
    _nd_uppr = df1[cond1]
    _nd_lower = df2[cond2]
    nd_overlap = pd.concat([_nd_lower, _nd_uppr],
                           axis=1).reindex(np.arange(ovr_lower, ovr_upp, 5)).interpolate('pad').fillna(0).dropna(
        how='all')

    nd_overlap['nd_res'] = nd_overlap[nd_overlap.columns[0]] * (
            ovr_upp - nd_overlap[nd_overlap.columns[0]].index.values) / \
                           (ovr_upp - ovr_lower) + nd_overlap[nd_overlap.columns[-1]] * \
                           (nd_overlap[nd_overlap.columns[-1]].index.values - ovr_lower) / (ovr_upp - ovr_lower)

    nd_overlap = nd_overlap.reindex(df1[cond1].index)
    # nd_overlap = nd_overlap.reindex(df2[cond2].index)
    return pd.concat([df1[df1.index <= ovr_lower], nd_overlap['nd_res'], df2[df2.index >= ovr_upp]])


def vel(d):
    return -0.1021 + 4.932 * d - 0.955 * d ** 2 + 0.07934 * d ** 3 - 0.0023626 * d ** 4


def pds_parameters(nd, d, dd):
    try:
        lwc = (np.pi / 6) * 1e-3 * np.sum(nd * d ** 3 * dd)  # g / m3
        dm = np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd)  # mm
        nw = 1e3 * (4 ** 4 / np.pi) * (lwc / dm ** 4)
        z = np.sum(nd * d ** 6 * dd)
        r = np.pi * 6e-4 * np.sum(nd * d ** 3 * vel(d) * dd)
        ref = ref_calc(nd=nd, d=d, dd=dd)
        return pd.Series([lwc, dm, nw, z, r]).append(ref)
    except ZeroDivisionError:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])



def bcksct(ds, instrument, ar=1, j=0) -> pd.DataFrame:
    """

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
    str_db = f"sqlite:///{path_db}/backscatter.sqlite"
    df_scatter.to_sql(f'{instrument}', con=str_db, if_exists='replace')
    return df_scatter


def ref_calc(nd, mie=False):
    ds = np.fromiter(nd.attrs['dsizes'].keys(), dtype=float) / 1e3
    try:
        path_db = f'{path_data}/db'
        make_dir(path_db)
        str_db = f"sqlite:///{path_db}/backscatter.sqlite"
        backscatter = pd.read_sql(f"{nd.attrs['instrument']}", con=str_db)
    except OperationalError:
        backscatter = bcksct(ds, nd.attrs['instrument'])
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


def get_attrs_merged(dt_attrs, _upper, _lower):
    d_2ds10 = np.fromiter(dt_attrs['2DS10']['dsizes'].keys(), dtype=float)
    d_hvps = np.fromiter(dt_attrs['HVPS']['dsizes'].keys(), dtype=float)
    idx_2ds10 = np.abs(d_2ds10 - _upper).argmin() + 1
    idx_hvps = np.abs(d_hvps - _upper).argmin()
    dt_2ds = dict(itertools.islice(dt_attrs['2DS10']['dsizes'].items(), idx_2ds10))
    dt_hvps = dict(itertools.islice(dt_attrs['HVPS']['dsizes'].items(), idx_hvps, len(d_hvps)))
    return {**dt_2ds, **dt_hvps}


def main():
    aircraft = 'Lear'
    _upper = 1200
    _lower = 800
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

    rdm_idx = pd.date_range(start='2019-09-07 2:31:50', periods=150, tz='UTC', freq='S')  # for Lear
    # rdm_idx = pd.date_range(start='2019-09-06 23:58:30', periods=60, tz='UTC', freq='S')  # for P3B
    # rdm_idx = df_concat.index
    indexx = rdm_idx

    attrs_merged = get_attrs_merged(df_concat.attrs, _upper, _lower)
    ls = []
    ls_z = []
    for i in indexx:
        df = df_concat.loc[i]
        res1 = df.unstack().T  # .reindex(index=np.arange(0.5, 10000, 0.25))
        if res1['2DS10'].dropna(how='any').empty:
            df1 = pd.Series(index=df_concat.attrs['2DS10']['bin_cent'], dtype='float64')
        else:
            df1 = res1['2DS10'].dropna(how='any')

        if res1['HVPS'].dropna(how='any').empty:
            df2 = pd.Series(index=df_concat.attrs['HVPS']['bin_cent'], dtype='float64')
        else:
            df2 = res1['HVPS'].dropna(how='any')

        a = linear_wgt(df1=df1, df2=df2, ovr_upp=_upper, ovr_lower=_lower)
        a.attrs['dsizes'] = attrs_merged
        a.attrs['instrument'] = 'Composite_PSD'
        z = ref_calc(a)
        ls_z.append(z)
        ls.append(a)

    df_reflectivity = pd.concat(ls_z, axis=1).T.set_index(indexx)
    df_merged = pd.concat(ls, axis=1).T.set_index(indexx)

    df_merged.attrs['dsizes'] = attrs_merged
    df_day = df_merged.groupby(df_merged.index.floor('d'))
    keys = list(df_day.groups.keys())
    del df_day

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
