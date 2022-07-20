#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        return lear_df
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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def g_func(xx):
    std = len(xx)
    mean = np.mean(xx)
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((xx - mean) / std) ** 2)


def get_convolution(array, half_window_size):
    array = np.concatenate((np.repeat(array[0], half_window_size),
                            array,
                            np.repeat(array[-1], half_window_size)))
    window_inds = [list(range(ind - half_window_size, ind + half_window_size + 1))
                   for ind in range(half_window_size, len(array) - half_window_size)]

    return np.take(array, window_inds)


def apply_smooth(x, y, hwz=20):
    x_conv = np.apply_along_axis(g_func, axis=1, arr=get_convolution(x, half_window_size=hwz))
    y_conv = get_convolution(y, half_window_size=hwz)
    y_mean = np.mean(y_conv, axis=1)
    y_centered = y_conv - y_mean[:, None]
    smoothed = np.sum(x_conv * y_centered, axis=1) / (hwz * 2) + y_mean
    return smoothed


def change_cols(df):
    # print(df.attrs['instrument'])
    attrs = df.attrs
    df = df.filter(like='nsd')
    bin_cent = df.attrs['bin_cent']
    cols = df.columns
    new_cols = {cols[i]: bin_cent[i] for i in range(len(cols))}
    df = df.rename(columns=new_cols)
    if df.empty:
        sr = pd.Series(index=bin_cent, name=f"psd_{df.attrs['instrument']}", dtype='float64')
        sr.attrs = attrs
        return sr
    else:
        sr = pd.Series(data=df.where(df > 0).T.iloc[1:, 0], name=f"{df.attrs['instrument']}")
        sr.attrs = attrs
        return sr


def sensitivity_analysis(sr, aircraft, idx):
    sr = sr.dropna()
    log_d = np.log10(sr.index).to_numpy().reshape((-1, 1))
    log_psd = np.ma.masked_invalid(np.log10(sr.values))

    xp = np.linspace(log_d.min(), log_d.max(), 300)
    yp = np.interp(xp, log_d.flatten(), log_psd.flatten())

    hwz = np.arange(25, 155, 20)
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.plot(log_d, log_psd, c='k', lw=0.5, label='mean PSD')
    ax2.plot(log_d, log_psd, c='k', lw=0.5, label='mean PSD')
    for i in hwz:
        smoothed_int = apply_smooth(x=xp, y=yp, hwz=i)
        yinterp = savgol_filter(yp, window_length=i, polyorder=3)
        ax2.plot(xp, smoothed_int, label=f'Gaussian {i}')
        ax1.plot(xp, yinterp, label=f'Savgol {i}')

    ax1.legend()
    ax1.set_xlabel('Log10(D)')
    ax1.set_ylabel('Log10 (N(D))')

    ax2.legend()
    ax2.set_xlabel('Log10(D)')
    ax2.set_ylabel('Log10 (N(D))')
    title = f"{idx: %Y-%m-%d %H:%M:%S} UTC - {aircraft}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.92)
    path_save = f'{path_data}/results/bimodality/sensitivity/{aircraft}'
    make_dir(path_save)
    fig.savefig(f"{path_save}/{aircraft}_{idx:%Y%m%d-%H%M%S}_sensitivity.jpg")


def check_bimodality2(sr, aircraft, idx):
    sr = sr.dropna()
    log_d = np.log10(sr.index).to_numpy().reshape((-1, 1))
    log_psd = np.ma.masked_invalid(np.log10(sr.values))

    xp = np.linspace(log_d.min(), log_d.max(), 300)
    yp = np.interp(xp, log_d.flatten(), log_psd.flatten())

    # smoothed_int = apply_smooth(x=xp, y=yp, hwz=35)
    model = LinearRegression().fit(xp.reshape(-1, 1), yp.reshape(-1, 1))
    y_pred_interp = model.intercept_ + model.coef_ * xp

    diff_interp = yp - y_pred_interp
    smoothed_res = apply_smooth(xp, diff_interp.flatten(), hwz=35)

    _idx_smt_res = np.argwhere(smoothed_res.flatten() >= 0)
    smoothed_res = smoothed_res[_idx_smt_res].flatten()
    xp_smt_res = xp[_idx_smt_res]

    yinterp = savgol_filter(yp, 51, 3)
    diff_savgol = yinterp - y_pred_interp.flatten()

    # _idx_savgol = np.argwhere(diff_savgol.flatten() >= 0)
    # diff_savgol = diff_savgol[_idx_savgol].flatten()
    # xp_dff_savgol = xp[_idx_savgol]
    #
    diff_savgol = diff_savgol
    xp_dff_savgol = xp

    # thresh_top = np.median(x) + 1 * np.std(x)
    #  Finding peaks
    # peaks_smooth, prop_smt = find_peaks(smoothed_res, height=0, prominence=0.0, width=10)
    peaks_savgol, prop_sav = find_peaks(diff_savgol, height=0, prominence=0.0, width=10)

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.plot(log_d, log_psd, c='k', lw=0.5, label='mean PSD')
    ax1.plot(xp, y_pred_interp.flatten(), label='Linear Reg.')
    # ax1.plot(xp, yp, label='Linear interpolation')
    # ax1.plot(xp, smoothed_int, label='Gaussian smoothing')
    ax1.plot(xp, yinterp, label='Savgol smoothing', c='red')

    ax1.legend()
    ax1.set_xlabel('Log10(D)')
    ax1.set_ylabel('Log10 (N(D))')

    ax2.plot(xp, diff_interp.flatten(), label='Residual from interp.')
    # ax2.plot(xp_smt_res, smoothed_res.flatten(), label='Smoothed Residuals')
    ax2.plot(xp_dff_savgol, diff_savgol, label='Savgol Residuals', c='red')
    # ax2.plot(xp_smt_res[peaks_smooth], smoothed_res.flatten()[peaks_smooth], "x", c='k', lw=1.3,
    #          label='Peaks smoothed res.')
    ax2.plot(xp_dff_savgol[peaks_savgol], diff_savgol.flatten()[peaks_savgol], "D", c='k', lw=1.3, label='Peaks Savgol')

    ax2.set_ylabel('Residuals')
    ax2.set_xlabel('Log10(D)')
    ax2.axhline(y=0, zorder=120, c='grey', lw=0.5)
    # ax2.vlines(x=xp_smt_res[peaks_smooth], ymin=0, ymax=smoothed_res.flatten()[peaks_smooth], color="k", ls='--')
    # ax2.hlines(y=prop_smt["width_heights"], xmin=xp_smt_res[prop_smt["left_ips"].astype(int)],
    #            xmax=xp_smt_res[prop_smt["right_ips"].astype(int)], color="k", ls='--', zorder=1)

    ax2.vlines(x=xp_dff_savgol[peaks_savgol], ymin=0, ymax=diff_savgol.flatten()[peaks_savgol], color="k", ls='--')
    ax2.hlines(y=np.zeros_like(xp_dff_savgol[prop_sav["left_ips"].astype(int)]),
               xmin=xp_dff_savgol[prop_sav["left_ips"].astype(int)],
               xmax=xp_dff_savgol[prop_sav["right_ips"].astype(int)], color="k", ls='--', zorder=1)

    ax2.legend()

    title = f"{idx: %Y-%m-%d %H:%M:%S} UTC - {aircraft}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.92)
    path_save = f'{path_data}/results/bimodality/fit_line/{aircraft}'
    make_dir(path_save)
    fig.savefig(f"{path_save}/{aircraft}_{idx:%Y%m%d-%H%M%S}_bimodality.jpg")


def plot_2d_dist(df_merged, keys, airc):
    for key in keys:
        df = df_merged.groupby(df_merged.index.floor('d')).get_group(key)
        df = df[df > 1.5]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        cbar = ax.pcolormesh(df.index, df.columns*1e-3, np.log10(df.T * 1e6), vmin=5, vmax=11, cmap='jet')
        plt.colorbar(cbar, pad=0.01, aspect=20)  # .set_ticks(np.arange(0,,1))
        ax.set_ylabel(r'$Diameter \  (mm)$', fontsize='x-large')
        ax.set_xlabel('$Time \  (UTC)$', fontsize='x-large')
        plt.title('$N(D), \log_{10} (m^{-3} mm^{-1}) $', position=(0.8, 0.85), fontsize='x-large')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        # ax.set_ylim(0, 200)
        ax.set_yscale('log')
        plt.show()
        print(1)


def plot_mean(_lear_df, idx, aircraft, limit=200):
    df_new = pd.concat(_lear_df, axis=1)
    df_new['mean'] = df_new.mean(axis=1).interpolate('pad', limit=limit)
    df_new.drop(columns=['psd_nan'], axis=1, inplace=True)
    plt.close('all')
    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    for i in _lear_df[:-1]:
        ax[0][0].step(x=i.index, y=i, where='post', label=i.name)
        ax[0][1].step(x=i.index, y=i, where='post', label=i.name)
    ax[0][1].step(x=df_new.index, y=df_new['mean'].values, where='pre', c='k', lw=1.8, label='mean')
    ax[0][0].set_yscale('log')
    ax[0][0].set_xscale('log')
    ax[0][0].set_xlabel('Particle size (um)')
    ax[0][0].set_ylabel('Concentration (# L-1 um-1)')
    ax[0][0].legend()
    ax[0][0].xaxis.grid(which='both')

    ax[0][1].set_yscale('log')
    ax[0][1].set_xscale('log')
    ax[0][1].set_xlabel('Particle size (um)')
    ax[0][1].set_ylabel('Concentration (# L-1 um-1)')
    ax[0][1].legend()
    ax[0][1].xaxis.grid(which='both')



    hawk = False
    if not hawk:
        _lear_df_wo = [i for i in _lear_df if not i.name.startswith('Hawk')]
    df_new_wo = pd.concat(_lear_df_wo, axis=1)
    df_new_wo['mean'] = df_new_wo.mean(axis=1).interpolate('pad', limit=limit)
    df_new_wo.drop(columns=['psd_nan'], axis=1, inplace=True)

    for i in _lear_df_wo[:-1]:
        ax[1][0].step(x=i.index, y=i, where='post', label=i.name)
        ax[1][1].step(x=i.index, y=i, where='post', label=i.name)
    ax[1][1].step(x=df_new_wo.index, y=df_new_wo['mean'].values, where='pre', c='k', lw=1.8, label='mean')
    ax[1][0].set_yscale('log')
    ax[1][0].set_xscale('log')
    ax[1][0].set_xlabel('Particle size (um)')
    ax[1][0].set_ylabel('Concentration (# L-1 um-1)')
    ax[1][0].legend()
    ax[1][0].xaxis.grid(which='both')

    ax[1][1].set_yscale('log')
    ax[1][1].set_xscale('log')
    ax[1][1].set_xlabel('Particle size (um)')
    ax[1][1].set_ylabel('Concentration (# L-1 um-1)')
    ax[1][1].legend()
    ax[1][1].xaxis.grid(which='both')

    title = f"{idx: %Y-%m-%d %H:%M:%S} UTC - {aircraft}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.92)
    path_save = f'{path_data}/results/bimodality/mean/{aircraft}'
    plt.show()
    # make_dir(path_save)
    # fig.savefig(f"{path_save}/{aircraft}_{idx:%Y%m%d-%H%M%S}_mean.jpg")


def main():
    # idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
    # idx = pd.Timestamp(year=2019, month=9, day=20, hour=2, minute=44, second=39, tz='UTC')
    # idx = pd.Timestamp(year=2019, month=9, day=20, hour=3, minute=16, second=58, tz='UTC')
    # idxs = find_idx(lear_df[:-1])
    limit = 100
    aircraft = 'Lear'
    ls_df = get_data(aircraft, temp=2)
    airc = ls_df[0].attrs['aircraft']
    rdm_idx = pd.date_range(start='2019-09-07 2:31:55', periods=120, tz='UTC', freq='S') # for Lear
    # rdm_idx = pd.date_range(start='2019-09-24 6:40:10', periods=120, tz='UTC', freq='S') # for p3b

    # rdm_idx = ls_df[-1]['totaln'].nlargest(1000).index
    ls_all = []
    # sr_mean = pd.Series(index=np.arange(1, 40000, 10), name='psd_nan', dtype='float16')
    for idx in rdm_idx:
        _ls_df = [change_cols(i.loc[i.index == idx]) for i in ls_df]
        hawk = False
        if not hawk:
            _ls_df = [i for i in _ls_df if not i.name.startswith('Hawk')]
        # _ls_df.append(sr_mean)
        # plot_mean(_ls_df, idx=idx, aircraft=ls_df[0].attrs['aircraft'], limit=limit)
        df_new = pd.concat(_ls_df, axis=1)
        df_new = df_new[df_new.index <= 40000]
        df_new[idx] = df_new[df_new > 0.1].mean(axis=1)#.interpolate('linear', limit=limit)
        ls_all.append(df_new[idx])

    df_merged = pd.concat(ls_all, axis=1).T
    df_day = df_merged.groupby(df_merged.index.floor('d'))
    keys = list(df_day.groups.keys())
    plot_2d_dist(df_merged, keys, airc)

    # idxs = ls_df[-1]['totaln'].nlargest(1000).index
    for idx in rdm_idx:
        _ls_df = [change_cols(i.loc[i.index == idx]) for i in ls_df]

        # Using all instruments
        hawk = True
        if not hawk:
            _ls_df = [i for i in _ls_df if not i.name.startswith('Hawk')]
        sr_mean = pd.Series(index=np.arange(1, 40000, 0.5), name='psd_nan', dtype='float64')
        _ls_df.append(sr_mean)
        plot_mean(_ls_df, idx=idx, aircraft=ls_df[0].attrs['aircraft'], limit=limit)

        # Using only Standalone instrument
        hawk = False
        if not hawk:
            _ls_df = [i for i in _ls_df if not i.name.startswith('Hawk')]
        sr_mean = pd.Series(index=np.arange(1, 40000, 0.5), name='psd_nan', dtype='float64')
        _ls_df.append(sr_mean)
        df_new = pd.concat(_ls_df, axis=1)
        df_new['mean'] = df_new[df_new > 0].mean(axis=1).interpolate('pad', limit=limit)
        df_new.drop(columns=['psd_nan'], axis=1, inplace=True)

        check_bimodality2(df_new['mean'], ls_df[0].attrs['aircraft'], idx)
        # sensitivity_analysis(df_new['mean'], ls_df[0].attrs['aircraft'], idx)


if __name__ == '__main__':
    main()
