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


def main():
    instruments = ['HVPS', '2DS10']
    ls_df_lear = [[pd.read_pickle(i) for i in ls_lear if instrument in i] for instrument in instruments]
    ls_df_lear = functools.reduce(operator.iconcat, ls_df_lear, [])
    idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    df_hvps = ls_df_lear[0]
    d = np.fromiter(df_hvps.attrs['dsizes'].keys(), dtype=float) / 1e3
    dd = np.fromiter(df_hvps.attrs['dsizes'].values(), dtype=float)
    cols = df_hvps.filter(like='nsd').columns
    df_hvps[['lwc', 'dm', 'nw', 'z']] = \
        df_hvps.apply(lambda x: pds_parameters(nd=(x.filter(like='nsd').values * 1e3), d=d, dd=dd), axis=1)
    df_hvps = df_hvps.dropna(subset=['lwc'])
    # df_hvps_norm = df_hvps.filter(like='nsd') * 1e3 / df_hvps['nw']
    # df_hvps_norm['dm'] = df_hvps['dm']
    print('entre a la grafica')
    fig, ax = plt.subplots()
    for index, row in df_hvps.iterrows():
        nd_norm = row[cols] * 1e3 / row['nw']
        d_dm = d / row['dm']
        ax.scatter(x=d_dm, y=nd_norm, c='black', linewidths=0.01)
    ax.set_yscale('log')
    ax.set_xlim(-1, 6)
    ax.set_ylim(1e-10, 1e1)
    ax.set_xlabel("D/Dm")
    ax.set_ylabel("N(D)/Nw")
    print('termine')
    plt.savefig('../results/hvps_norm_2.jpg')
    plt.show()
    pass


if __name__ == '__main__':
    main()
