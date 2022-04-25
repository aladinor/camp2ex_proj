#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
from re import split
import seaborn as sns
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def main():
    sensor = ['2DS10', 'Hawk2DS10']
    ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
    lear_df = [pd.read_pickle(i) for i in ls_lear]
    ls_df = [i for i in lear_df if i.attrs['type'] in sensor]
    dates = ls_df[0].index.intersection(ls_df[1].index)
    ds10 = ls_df[0].loc[dates].filter(like='nsd')
    ds10['local_time'] = ls_df[0]['local_time'].loc[dates]
    hawkds10 = ls_df[1].loc[dates].filter(like='nsd')
    hawkds10['local_time'] = ls_df[1]['local_time'].loc[dates]
    diff = ds10 - hawkds10
    diff['local_time'] = ds10['local_time']
    idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    sept = diff.groupby(by=diff['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))
    df_nd = sept
    plt.figure(figsize=(15, 4))
    a = plt.pcolormesh(df_nd.index.values, df_nd.columns[:-1], np.log10(df_nd[df_nd.columns[:-1]].T), cmap='seismic')
    plt.colorbar(a)
    # plt.pcolormesh(diff[diff.columns[:-1]].T)
    # plt.imshow(diff[diff.columns[:-1]].T, aspect='auto')

    # ax = sns.scatterplot(x=ds10[ds10.columns[0]], y=hawkds10[hawkds10.columns[0]])
    # ax1 = sns.boxplot(data=diff[diff.columns[15:35]])
    # ax1.set(yscale="log")
    # ax1.set_xlim(-10e3, 10e3)
    # ax1.set_ylim(-10e3, 10e3)
    plt.show()
    print(1)
    pass


if __name__ == '__main__':
    main()
