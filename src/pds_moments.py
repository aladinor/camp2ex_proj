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
import matplotlib
matplotlib.use('agg')

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']
ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')


def vel(d):
    return -0.1021 + 4.932 * d -0.955 * d ** 2 + 0.07934 * d ** 3 - 0.0023626 * d ** 4


def pds_parameters(nd, d, dd):
    try:
        lwc = (np.pi / 6) * 1e-3 * np.sum(nd * d ** 3 * dd)  # g / m3
        dm = np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd)  # mm
        nw = 1e3 * (4 ** 4 / np.pi) * (lwc / dm ** 4)
        z = np.sum(nd * d ** 6 * dd)
        r = np.pi * 6e-4 * np.sum(nd * d ** 3 * vel(d) * dd)
        return pd.Series([lwc, dm, nw, z, r])
    except ZeroDivisionError:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])


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
    ax1.set_xlim(-1, 14)
    ax1.set_ylim(1e-16, 1e3)
    ax1.set_ylabel("N(D)/Nw")
    ax1.set_title(f"{df.attrs['instrument']} on {df.attrs['aircraft']}")
    cbar = plt.colorbar(sc1, ax=ax1)
    cbar.set_label('Scaled DSD - N(D) / NW )')
    cbar.formatter.set_powerlimits((0, 0))
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
    for k in [ls_p3, ls_lear]:
        ls_df_lear = [[pd.read_pickle(i) for i in k if instrument in i] for instrument in instruments]
        ls_df_lear = functools.reduce(operator.iconcat, ls_df_lear, [])
        attrs = [i.attrs for i in ls_df_lear]

        if attrs[0]['aircraft'] == 'Learjet':
            ls_df_lear = [pd.merge(i, ls_df_lear[-1][["Temp"]], left_index=True, right_index=True) for i in ls_df_lear[:-1]]
        else:
            p3_merged = glob.glob(f'{path_data}/data/01_SECOND.P3B_MRG/MERGE/all/*pkl')
            p3_temp = pd.read_pickle(p3_merged[0])
            ls_df_lear = [pd.merge(i, p3_temp[' Static_Air_Temp_YANG_MetNav'], left_index=True, right_index=True)
                          for i in ls_df_lear]

        for i, atrs in enumerate(attrs[:-1]):
            ls_df_lear[i].attrs = atrs
            if ls_df_lear[i].attrs['aircraft'] == 'P3B':
                ls_df_lear[i].rename(columns={' Static_Air_Temp_YANG_MetNav': 'Temp'}, inplace=True)

        aircraft = ls_df_lear[0].attrs['aircraft']
        print(aircraft)
        df_hvps = ls_df_lear[0]
        cols = df_hvps.filter(like='nsd').columns
        df_hvps.loc[:, cols] = df_hvps[cols].mul(1e3)
        d_hvps = np.fromiter(df_hvps.attrs['dsizes'].keys(), dtype=float) / 1e3
        dd_hvps = np.fromiter(df_hvps.attrs['dsizes'].values(), dtype=float)
        df_hvps = df_hvps[df_hvps['Temp'] >= 2]
        df_hvps[['lwc', 'dm', 'nw', 'z', 'r']] = \
            df_hvps.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values, d=d_hvps, dd=dd_hvps), axis=1)
        df_hvps = df_hvps.dropna(subset=['lwc'])
        df_hvps.to_pickle(f"../results/df_{df_hvps.attrs['instrument']}_{df_hvps.attrs['aircraft']}_norm.pkl")
        plot_norm(df_hvps, d_hvps)

        df_2ds = ls_df_lear[1]
        cols = df_2ds.filter(like='nsd').columns
        df_2ds.loc[:, cols] = df_2ds[cols].mul(1e3)
        d_2ds10 = np.fromiter(df_2ds.attrs['dsizes'].keys(), dtype=float) / 1e3
        dd_2ds10 = np.fromiter(df_2ds.attrs['dsizes'].values(), dtype=float)
        df_2ds = df_2ds[df_2ds['Temp'] >= 2]
        df_2ds[['lwc', 'dm', 'nw', 'z', 'r']] = \
            df_2ds.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values, d=d_2ds10, dd=dd_2ds10), axis=1)
        df_2ds = df_2ds.dropna(subset=['lwc'])
        df_2ds.to_pickle(f"../results/df_{df_2ds.attrs['instrument']}_{df_2ds.attrs['aircraft']}_norm.pkl")
        plot_norm(df_2ds, d_2ds10)

        df_merged = merge_df(df_2ds10=df_2ds, df_hvps=df_hvps)
        df_merged = df_merged[df_merged['Temp'] >= 2]
        d_merged = np.fromiter(df_merged.attrs['dsizes'].keys(), dtype=float) / 1e3
        dd_merged = np.fromiter(df_merged.attrs['dsizes'].values(), dtype=float)
        df_merged[['lwc', 'dm', 'nw', 'z', 'r']] = \
            df_merged.apply(lambda x: pds_parameters(nd=(x.filter(like='nsd').values * 1e3), d=d_merged, dd=dd_merged), axis=1)
        plot_norm(df_merged, d_merged)
        df_merged.to_pickle(f"../results/df_{df_merged.attrs['instrument']}_{df_merged.attrs['aircraft']}_norm.pkl")


if __name__ == '__main__':
    main()
