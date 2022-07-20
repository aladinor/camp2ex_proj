#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from IPython import display
import functools
import operator
from dask import delayed, compute
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.animation import ArtistAnimation, FFMpegWriter
from re import split
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from matplotlib import use

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

# use('Agg')

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def get_data(instrument='Lear', temp=2):
    """

    :param instrument: aircraft
    :param temp: temperature for filtering
    :return: list of available dataframe in camp2ex
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


def plot_mean(df_new, idx, aircraft, instr):
    df_new = df_new[df_new > 0]
    # plt.close('all')
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 7))
    for i in df_new.columns[:-1]:
        ax.step(x=df_new.index.values * 1e-3, y=df_new[i] * 1e6, where='post', label=i)
        ax1.step(x=df_new.index.values * 1e-3, y=df_new[i] * 1e6, where='post', label=i)
    ax1.step(x=df_new.index * 1e-3, y=df_new['mean'].values * 1e6, where='post', c='k', lw=1.8, label='mean')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Particle size (um)')
    ax.set_ylabel('Concentration (# L-1 um-1)')
    ax.legend()
    ax.xaxis.grid(which='both')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Particle size (um)')
    ax1.set_ylabel('Concentration (# L-1 um-1)')
    ax1.legend()
    ax1.xaxis.grid(which='both')
    title = f"{idx: %Y-%m-%d %H:%M:%S} UTC - {aircraft}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.92)
    path_save = f'{path_data}/results/bimodality/mean/{aircraft}'
    plt.show()
    # make_dir(path_save)
    # fig.savefig(f"{path_save}/{aircraft}_{idx:%Y%m%d-%H%M%S}_mean.jpg")


def series_df(df):
    # print(df.attrs['instrument'])
    attrs = df.attrs
    bin_cent = df.attrs['bin_cent']
    if len(df.index) == 0:
        sr = dd.from_pandas(pd.Series(index=bin_cent, name=f"psd_{df.attrs['instrument']}", dtype='float64'),
                            npartitions=5)
        sr.attrs = attrs
        return sr
    else:
        sr = dd.from_pandas(pd.Series(data=df.where(df > 0).compute().values[0, :], index=bin_cent,
                                      name=f"{df.attrs['instrument']}"), npartitions=1)
        sr.attrs = attrs
        return sr


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
    cols = [[j for j in i.columns if j.startswith('nsd')] for i in ls_df]
    ls_df = [i[cols[j]] for j, i in enumerate(ls_df)]
    ls_df = [change_cols(i) for i in ls_df]
    return ls_df


def main():
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
    limit = 500
    aircraft = 'Lear'
    ls_df = get_data(aircraft, temp=2)
    ls_df = filt_by_instrument(ls_df)
    ls_df = filt_by_cols(ls_df)
    ls_df = compute(*ls_df)
    instr = [i.attrs['instrument'] for i in ls_df]

    a = pd.concat(compute(*ls_df), axis=1, keys=instr, levels=[instr])

    rdm_idx = pd.date_range(start='2019-09-07 2:31:45', periods=150, tz='UTC', freq='S') # for Lear
    # rdm_idx = pd.date_range(start='2019-09-06 23:58:30', periods=60, tz='UTC', freq='S')  # for P3B
    indexx = rdm_idx
    # indexx = a.index

    sr_mean = pd.Series(index=np.arange(0.5, 10000, 0.5), name='psd_nan', dtype='float16')
    ls = []
    for i in indexx:
        df = a.loc[i]
        res1 = df.unstack().T#.reindex(index=np.arange(0.5, 10000, 0.25))
        res1 = pd.concat([res1, sr_mean], axis=1)
        # res1 = res1.apply(lambda x: x.interpolate('pad', limit=100), axis=0)
        res1['mean'] = res1[res1 > 0].mean(axis=1, skipna=True)
        # plot_mean(res, idx=i, aircraft=aircraft, instr=instr)
        ls.append(res1['mean'])

    df_merged = pd.concat(ls, axis=1).T.set_index(indexx)

    # fig = plt.figure(figsize=(15, 8))
    # gs = GridSpec(2, 4, figure=fig)
    # ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax3 = fig.add_subplot(gs[1, 1])
    # ax4 = fig.add_subplot(gs[1, 2])
    # ax5 = fig.add_subplot(gs[1, 3])
    #
    # artist = []
    # for i in df_merged.index:
    #     ax1.clear()
    #     ax2.clear()
    #     ax3.clear()
    #     df_plot = df_merged.copy()
    #     df_plot[~(df_plot.index <= i)] = np.nan
    #     im1 = ax1.pcolormesh(df_plot.index, df_plot.columns * 1e-3,
    #                          np.log10(df_plot.T * 1e6), vmin=5, vmax=11, cmap='jet')
    #     df = a.loc[i].copy(deep=True)
    #     ls_ = [df[k] for k in instr]
    #     del df
    #     for j in range(len(ls_)):
    #         ls_[j].name = instr[j]
    #     df_new = pd.concat(ls_, axis=1)
    #     del ls_
    #     df_new = df_new[df_new > 0]
    #     df_new['mean'] = df_new.mean(axis=1, skipna=True)  # .interpolate('pad', limit=250)
    #     im2, = ax2.step(x=df_new.index.values * 1e-3, y=df_new['FCDP'] * 1e6, where='post', label='FCDP', c='r')
    #     im3, = ax3.step(x=df_new.index.values * 1e-3, y=df_new['2DS10'] * 1e6, where='post', label='2DS10', c='b')
    #     im4, = ax4.step(x=df_new.index.values * 1e-3, y=df_new['HVPS'] * 1e6, where='post', label='HVPS', c='g')
    #     im5, = ax5.step(x=df_new.index * 1e-3, y=df_new['mean'].values * 1e6, where='post',
    #                     c='k', lw=1.8, label='mean')
    #     ax2.set_ylim(df_new.min().min()*1e6, df_new.max().max()*1e6)
    #     ax3.set_ylim(df_new.min().min()*1e6, df_new.max().max()*1e6)
    #     ax4.set_ylim(df_new.min().min()*1e6, df_new.max().max()*1e6)
    #     ax5.set_ylim(df_new.min().min()*1e6, df_new.max().max()*1e6)
    #
    #     ax2.set_xlim(df_new.index.min()*1e-3, 10)
    #     ax3.set_xlim(df_new.index.min()*1e-3, 10)
    #     ax4.set_xlim(df_new.index.min()*1e-3, 10)
    #     ax5.set_xlim(df_new.index.min()*1e-3, 10)
    #     artist.append([im1, im2, im3, im4, im5])
    #
    # ax2.legend(['FCDP'])
    # ax3.legend(['2DS10'])
    # ax4.legend(['HVPS'])
    # ax5.legend(['MEAN'])
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xscale('log')
    # ax3.set_yscale('log')
    # ax3.set_xscale('log')
    # ax2.set_xlabel('Particle size (mm)')
    # ax2.set_ylabel('Concentration (# m-3 mm-1)')
    # ax2.xaxis.grid(which='both')
    # ax3.set_xlabel('Particle size (mm)')
    # ax3.xaxis.grid(which='both')
    # ax1.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
    # ax1.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
    # ax1.set_title('$N(D), \log_{10} (\# \ mm^{-3} m^{-1}) $', position=(0.8, 0.1), fontsize='x-large')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # ax4.set_yscale('log')
    # ax5.set_xscale('log')
    # ax4.set_xscale('log')
    # ax5.set_yscale('log')
    # ax4.xaxis.grid(which='both')
    # ax5.xaxis.grid(which='both')
    # ax4.set_xlabel('Particle size (mm)')
    # ax5.set_xlabel('Particle size (mm)')
    # plt.colorbar(im1, ax=ax1, pad=0.01, aspect=20)
    #
    # anim = ArtistAnimation(fig, artist, interval=100, blit=True)
    # writervideo = FFMpegWriter(fps=60)
    # anim.save('../results/lear.mp4', writer=writervideo)
    df_day = df_merged.groupby(df_merged.index.floor('d'))
    keys = list(df_day.groups.keys())
    del df_day

    for key in keys:
        airc = ls_df[0].attrs['aircraft']
        df = df_merged.groupby(df_merged.index.floor('d')).get_group(key)
        df = df[df > 0]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        cbar = ax.pcolormesh(df.index, df.columns * 1e-3, np.log10(df.T * 1e6), vmin=0, vmax=10, cmap=my_cmap)
        plt.colorbar(cbar, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        ax.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
        ax.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
        plt.title('$N(D), \log_{10} (\# \ m^{-3} mm^{-1}) $', position=(0.8, 0.1), fontsize='x-large')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_yscale('log')
        title = f"{key: %Y-%m-%d} UTC - {aircraft}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
        # path_save = f'{path_data}/results/bimodality/flight/{aircraft}'
        # make_dir(path_save)
        # fig.savefig(f"{path_save}/{aircraft}_{key:%Y%m%d}.jpg")
        plt.show()
        print(1)


if __name__ == '__main__':
    main()
