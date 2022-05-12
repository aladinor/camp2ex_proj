#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
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


def pds_parameters(nd_serie):
    nd = nd_serie.values
    d = np.fromiter(nd_serie.attrs['dsizes'].keys(), dtype=float)
    dd = np.fromiter(nd_serie.attrs['dsizes'].values(), dtype=float)
    lwc = (np.pi / 6) * 10e-9 * np.sum(nd * d ** 3 * dd)
    dm = 10e-3 * (np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd))
    nw = 1*10e9 * (4 ** 4 / (np.pi * 1000)) * (lwc / dm ** 4)
    z = 10e-15 * np.sum(nd * d ** 6 * dd)
    return lwc, dm, nw, z


def main():
    instruments = ['HVPS', '2DS10']
    ls_df_lear = [[pd.read_pickle(i) for i in ls_lear if instrument in i] for instrument in instruments]
    ls_df_lear = functools.reduce(operator.iconcat, ls_df_lear, [])
    idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    df_hvps = ls_df_lear[0]
    nd_series = df_hvps.loc[df_hvps['local_time'] == idx].filter(like='nsd')
    a = pds_parameters(nd_series)
    pass


if __name__ == '__main__':
    main()
