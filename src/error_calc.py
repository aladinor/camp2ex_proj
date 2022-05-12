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
from matplotlib.dates import MinuteLocator, HourLocator, DateFormatter
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def multiple_plots(df, air):
    df['utc'] = df['local_time'].tz_convert(tz='UTC')
    df.drop(columns=['local_time'], inplace=True)
    days = df.groupby(by=df['utc'].dt.floor('d'))
    ncols = 4
    nrows = int(np.ceil(days.ngroups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 9), sharey=True)
    axs = axes.flatten()
    i = 0
    for _time, group in days:
        a = axs[i].pcolormesh(group.index.values, group.columns[:-1], np.log10(group[group.columns[:-1]].T),
                              cmap='seismic',
                              shading='auto',
                              vmin=-3, vmax=3
                              )
        axs[i].set_title(f'{_time:%Y-%m-%d}')
        axs[i].xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 1)))
        axs[i].xaxis.set_minor_locator(MinuteLocator(interval=15))
        axs[i].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        y_labels = [tick.split('-')[-1] for tick in group.filter(like='nsd').columns]
        y_labels = ['' if i % 5 else tick for i, tick in enumerate(y_labels)]
        y_labels[-1] = y_labels[-1].split('>')[-1]
        axs[i].set_yticklabels(y_labels, fontsize=7)
        axs[i].tick_params(axis='x', labelsize=7)
        i += 1
    fig.supxlabel("Time", fontsize=12)
    fig.supylabel("Diameter (um)", fontsize=12)
    fig.suptitle(f"2DS10 - Hawk2DS10 ({air})",  fontsize=14)
    plt.tight_layout()
    fig.colorbar(a, label="Concentration (# L um-1 )", ax=axs, pad=0.02, aspect=50)
    plt.savefig(f'../results/multiple_diff_2ds10_{air}.jpg')
    # plt.show()
    print(1)


def single_plot(df_nd, air, idx):
    fig, ax = plt.subplots(figsize=(15, 4))
    a = plt.pcolormesh(df_nd.index.values, df_nd.columns[:-1], np.log10(df_nd[df_nd.columns[:-1]].T), cmap='seismic',
                       shading='auto',
                       vmin=-3, vmax=3
                       )
    fig.colorbar(a, label="Concentration (# L um-1 )", pad=0.02, aspect=50)
    ax.set_xlabel("Time")
    ax.set_ylabel('Diameter (um)')
    ax.set_title(f'{idx:%Y-%m-%d}')
    y_labels = [tick.split('-')[-1] for tick in df_nd.filter(like='nsd').columns]
    y_labels = [tick if i % 4 else '' for i, tick in enumerate(y_labels)]
    y_labels = ['' if i % 2 else tick for i, tick in enumerate(y_labels)]
    ax.set_yticklabels(y_labels)
    fig.suptitle(f"2DS10 - Hawk2DS10 ({air})", fontsize=14)
    plt.savefig(f'../results/single_diff_2ds10_{air}.jpg')
    print(1)


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
    diff = ds10.filter(like='nsd') - hawkds10.filter(like='nsd')
    diff['local_time'] = ds10['local_time']
    idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    sept = diff.groupby(by=diff['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))

    single_plot(sept, ds10.attrs['aircraft'], idx)
    # multiple_plots(diff, ds10.attrs['aircraft'])

    sensor = ['2DS10', 'Hawk2DS10']
    ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
    lear_df = [pd.read_pickle(i) for i in ls_lear]
    ls_df = [i for i in lear_df if i.attrs['type'] in sensor]
    dates = ls_df[0].index.intersection(ls_df[1].index)
    ds10 = ls_df[0].loc[dates].filter(like='nsd')
    ds10['local_time'] = ls_df[0]['local_time'].loc[dates]
    hawkds10 = ls_df[1].loc[dates].filter(like='nsd')
    hawkds10['local_time'] = ls_df[1]['local_time'].loc[dates]
    diff_2 = ds10.filter(like='nsd') - hawkds10.filter(like='nsd')
    diff_2['local_time'] = ds10['local_time']
    idx_2 = pd.Timestamp(year=2019, month=9, day=17, hour=10, minute=32, second=21, tz='Asia/Manila')
    sept_2 = diff_2.groupby(by=diff_2['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx_2.date(), tz='Asia/Manila'))
    #
    single_plot(sept_2, ds10.attrs['aircraft'], idx)
    # multiple_plots(diff_2, ds10.attrs['aircraft'])
    # pass


if __name__ == '__main__':
    main()
