#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import functools
import operator
import pandas as pd
from re import split
from datetime import datetime

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']
ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')


def pds_parameters(nd, d, dd):
    try:
        lwc = (np.pi / 6) * 1e-3 * np.sum(nd * d ** 3 * dd)  # g / m3
        dm = np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd)  # mm
        nw = 1e3 * (4 ** 4 / np.pi) * (lwc / dm ** 4)
        z = np.sum(nd * d ** 6 * dd)
        return pd.Series([lwc, dm, nw, z])
    except ZeroDivisionError:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])


def plot_norm(df, d):
    start_time = datetime.now()
    print(df.attrs['aircraft'])
    cols = df.filter(like='nsd').columns
    fig, ax1 = plt.subplots(figsize=(8, 6))
    for index, row in df.iterrows():
        nd_norm = row[cols] / row['nw']
        d_dm = d / row['dm']
        sc1 = ax1.scatter(x=d_dm, y=nd_norm, s=0.01, c=nd_norm)
    ax1.set_yscale('log')
    ax1.set_xlabel("D/Dm")
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(1e-12, 1e2)
    ax1.set_ylabel("N(D)/Nw")
    ax1.set_title(f"{df.attrs['instrument']} on {df.attrs['aircraft']}")
    plt.colorbar(sc1, ax=ax1)
    plt.savefig(f"../results/{df.attrs['aircraft']}_{df.attrs['instrument']}.jpg", dpi=400)
    plt.close('all')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


def plot_psd(psd1, psd1_d, psd2, psd2_d, _idx, aircraft):
    fig, ax = plt.subplots()
    psd1 = np.where(psd1[0] > 0, psd1[0], np.nan)
    psd2 = np.where(psd2[0] > 0, psd2[0], np.nan)
    ax.step(psd1_d, psd1, label='2DS10')
    ax.step(psd2_d, psd2, label='HVPS')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.grid(which='both')
    ax.set_title(f"{_idx: %Y-%m-%d %H:%M:%S} (UTC) - {aircraft}")
    ax.set_xlabel(f"$Diameter \ (\mu m)$")
    ax.set_ylabel("$Concentration \ (\#  L^{-1} \mu m^{-1})$")
    plt.show()


def merge_df(df_2ds10, df_hvps):
    cols1 = df_2ds10.filter(like='nsd').columns[:-16] # up to 1 mm
    attrs1 = df_2ds10.attrs
    attrs2 = df_hvps.attrs
    cols2 = df_hvps.filter(like='nsd').columns[6:]
    df_final = pd.merge(df_2ds10[cols1], df_hvps[cols2], right_index=True, left_index=True)
    df_final = pd.merge(df_final, df_hvps['Temp'], right_index=True, left_index=True)
    sizes = np.concatenate((attrs1['sizes'][:-16], attrs2['sizes'][6:]))
    dsizes = list(attrs1['dsizes'].items())[:-16] + list(attrs2['dsizes'].items())[6:]
    dsizes = dict(dsizes)
    attrs_final = attrs1
    attrs_final['sizes'] = sizes
    attrs_final['dsizes'] = dsizes
    attrs_final['instrument'] = attrs1['instrument'] + '-' + attrs2['instrument']
    df_final.attrs = attrs_final
    return df_final


def main():
    instruments = ['HVPS', '2DS10', 'Page0']
    ls_df_lear = [[pd.read_pickle(i) for i in ls_lear if instrument in i] for instrument in instruments]
    ls_df_lear = functools.reduce(operator.iconcat, ls_df_lear, [])
    attrs = [i.attrs for i in ls_df_lear]

    ls_df_lear = [pd.merge(i, ls_df_lear[-1][["Temp"]], left_index=True, right_index=True) for i in ls_df_lear[:-1]]
    for i, atrs in enumerate(attrs[:-1]):
        ls_df_lear[i].attrs = atrs

    aircraft = ls_df_lear[0].attrs['aircraft']
    df_hvps = ls_df_lear[0]
    cols = df_hvps.filter(like='nsd').columns
    df_hvps[cols] = df_hvps[cols].mul(1e3)
    d_hvps = np.fromiter(df_hvps.attrs['dsizes'].keys(), dtype=float) / 1e3
    dd_hvps = np.fromiter(df_hvps.attrs['dsizes'].values(), dtype=float)
    df_hvps = df_hvps[df_hvps['Temp'] >= 2]
    df_hvps[['lwc', 'dm', 'nw', 'z']] = \
        df_hvps.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values, d=d_hvps, dd=dd_hvps), axis=1)
    df_hvps = df_hvps.dropna(subset=['lwc'])
    plot_norm(df_hvps, d_hvps)

    df_2ds = ls_df_lear[1]
    d_2ds10 = np.fromiter(df_2ds.attrs['dsizes'].keys(), dtype=float) / 1e3
    dd_2ds10 = np.fromiter(df_2ds.attrs['dsizes'].values(), dtype=float)
    df_2ds = df_2ds[df_2ds['Temp'] >= 2]
    df_2ds[['lwc', 'dm', 'nw', 'z']] = \
        df_2ds.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values, d=d_2ds10, dd=dd_2ds10), axis=1)
    df_2ds = df_2ds.dropna(subset=['lwc'])
    plot_norm(df_2ds, d_2ds10)

    df_merged = merge_df(df_2ds10=df_2ds, df_hvps=df_hvps)
    df_merged = df_merged[df_merged['Temp'] >= 2]
    d_merged = np.fromiter(df_merged.attrs['dsizes'].keys(), dtype=float) / 1e3
    dd_merged = np.fromiter(df_merged.attrs['dsizes'].values(), dtype=float)
    df_merged[['lwc', 'dm', 'nw', 'z']] = \
        df_merged.apply(lambda x: pds_parameters(nd=(x.filter(like='nsd').values * 1e3), d=d_merged, dd=dd_merged), axis=1)
    plot_norm(df_merged, d_merged)

    # df_merged.to_csv('../results/df_merged_norm.csv')
    # df_merged = pd.read_csv('../results/df_merged_norm.csv')
    # # print('entre a la grafica')


    #
    #
    # idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    #
    # df_hvps = ls_df_lear[0]
    # # df_hvps[['lwc', 'dm', 'nw', 'z']] = \
    # #     df_hvps.apply(lambda x: pds_parameters(nd=(x.filter(like='nsd').values * 1e3), d=d_hvps, dd=dd_hvps), axis=1)
    # # df_hvps = df_hvps.dropna(subset=['lwc'])
    # # df_hvps = df_hvps[df_hvps['Temp'] >= 2]
    # # df_hvps.to_csv('../results/df_hvps_norm.csv')
    # # df_hvps = pd.read_csv('../results/df_hvps_norm.csv')
    #
    # df_2ds = ls_df_lear[1]
    # # nd_test_hvps = df_hvps.loc[df_hvps['local_time'] == idx].filter(like='nsd').values * 1e3
    # # nd_test_2ds10 = ls_df_lear[1].loc[ls_df_lear[1]['local_time'] == idx].filter(like='nsd').values * 1e3
    #
    # # plot_psd(psd1=nd_test_2ds10, psd1_d=d_2ds10, psd2=nd_test_hvps,
    # #          psd2_d=d_hvps, _idx=idx, aircraft=aircraft)
    #
    # # pds_parameters(nd_test_hvps, d_hvps, dd_hvps)

    print('termine')
    pass


if __name__ == '__main__':
    main()
