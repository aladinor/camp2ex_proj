#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.constants import c
from scipy.special import gamma
from scipy.optimize import minimize, brentq, newton
import pandas as pd
from dask import delayed, compute
from re import split

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini
from src.radar_utils import bck_extc_crss

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def integral(dm, d, dd, mu=3, instrument='Composite_PSD', mie=False, band='Ku'):
    if mie is False:
        sct = f'T_mat_{band}'
    else:
        sct = f'Mie_{band}'
    bsc = bck_extc_crss(d.values, instrument=instrument)
    sigma_b = xr.DataArray(data=bsc[sct],
                           dims=['diameter'],
                           coords=dict(diameter=(["diameter"], d.diameter.values)))
    f_mu = (6 * (mu + 4) ** (mu + 4)) / (4 ** 4 * gamma(mu + 4))
    i_b = sigma_b * f_mu * (d / dm) ** mu * np.exp(-(mu + 4) * (d / dm)) * dd
    return i_b.sum('diameter')


def objective_func(dm, xr_comb, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return ku - ka


def eq_funct(dm, xr_comb, mu=3, dfr=None):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    if dfr is None:
        dfr = xr_comb.dbz_t_ku - xr_comb.dbz_t_ka
    return dfr - ku + ka


def dm_solver(xr_comb, mu=3):
    dm_bounds = [[0.1, 3.5]]
    dm_const = {'type': 'eq', 'fun': eq_funct, 'args': (xr_comb, mu)}
    dm_0 = np.array([1.75])
    res = minimize(fun=objective_func, x0=dm_0, args=(xr_comb, mu), bounds=dm_bounds, constraints=dm_const, tol=1e-2,
                   options={'maxiter': 50})
    return xr.DataArray(res['x'], dims=['time'], coords=dict(time=(['time'], np.array([xr_comb.time.values]))))


def dm_retrieval(dm1, dm2, xr_comb, mu=3, tol=0.00001, maxiter=100):
    _iter = 0
    ku_ka = objective_func(xr_comb.dm, xr_comb, xr_comb.mu).values
    for i in range(maxiter):
        fx1 = eq_funct(dm1, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        fx2 = eq_funct(dm2, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        dm_new = dm1 - fx1 * (dm2 - dm1) / (fx2 - fx1)
        diff = np.abs(dm_new-dm2)
        if np.isnan(diff):
            return np.nan
        dm2 =dm1
        dm1 = dm_new
        _iter += 1
        if tol > diff:
            return dm1


def dm_plt(dm1, dm2, xr_comb, mu=3, tol=0.00000001, maxiter=100):
    diff = 1
    _iter = 0
    dm_s = []
    f_res = []
    f_res2 = []
    x = np.arange(0.1, 5, 0.1)
    ku_ka = objective_func(xr_comb.dm, xr_comb, xr_comb.mu).values
    dm_plot = np.array([eq_funct(i, xr_comb=xr_comb, mu=mu, dfr=ku_ka) for i in x])
    dm_plot2 = np.array([eq_funct(i, xr_comb=xr_comb, mu=mu) for i in x])
    dfr = (xr_comb.dbz_t_ku - xr_comb.dbz_t_ka).values

    dfr_ib = dfr - ku_ka
    fig, ax = plt.subplots(dpi=180)
    ax.plot(x, dm_plot, c='orange', zorder=1)
    ax.plot(x, dm_plot2, c='orange', zorder=1)
    ax.scatter(xr_comb.dm, dfr_ib, label='True Dm', zorder=2)
    ax.set_xlabel('Dm  (mm)')
    ax.set_ylabel(f'DFR - Ib(Ku) + Ib(Ka) (dB)')
    ax.legend()
    title = f"{pd.to_datetime(xr_comb.time.values): %Y-%m-%d %X} - UTC"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    ax.grid()
    ax.text(3, 0, r'$D_m$' + f"={xr_comb.dm.values:.2f}")
    ax.text(3, 0.6, r'$DFR$' + f"={ku_ka:.2f}")
    ax.text(3, -0.6, r'$\mu$' + f"={xr_comb.mu.values:.2f}")
    # plt.show()
    print(1)
    while not (_iter > maxiter or tol > diff):
        fx1 = eq_funct(dm1, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        fx2 = eq_funct(dm2, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        dm_new = dm1 - fx1 * (dm2 - dm1) / (fx2 - fx1)
        diff = np.abs(dm_new-dm2)
        if np.isnan(diff):
            return np.nan
        dm2 = dm1
        dm1 = dm_new
        dm_s.append(dm1)
        f_res.append(fx1)
        f_res2.append(fx2)
        _iter += 1
        ax.scatter(dm_new, fx1, c='k', s=0.8, label='fx1', zorder=10)
        # ax.scatter(dm1, fx2, c='k', s=0.8, label='fx2', zorder=10)
        # plt.show()
        print(1)
    ax.text(3, -1.2, r'$D_{m_{est}}$' + f"={dm1:.2f}")
    print(1)
    return dm1


def rain_rate():
    rr = 6 * np.pi * 1e-4 * nw
    return


def main():
    xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_Lear_300_1000_5_bins_merged.zarr')
    xr_comb = xr_comb.isel(time=range(15, 25))
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    # individal test
    test_dm = dm_plt(dm1=0.5, dm2=3.5, xr_comb=xr_comb.isel(time=0), mu=xr_comb.isel(time=0).mu.values)
    dm_sol = [dm_retrieval(dm1=0.5, dm2=3.5, xr_comb=i.load(), mu=i.mu.load())
              for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]

    dm_gpm = [dm_retrieval(dm1=0.5, dm2=3.5, xr_comb=i.load(), mu=3)
              for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]

    xr_comb['dm_dfr'] = xr.DataArray(np.array(dm_sol),
                                     dims=['time'],
                                     coords=dict(time=(['time'], xr_comb.time.values)))

    xr_comb['dm_gpm'] = xr.DataArray(np.array(dm_gpm),
                                     dims=['time'],
                                     coords=dict(time=(['time'], xr_comb.time.values)))

    path_save = f"{path_data}/cloud_probes/zarr"
    _ = xr_comb[['dm', 'dm_dfr', 'dm_gpm']].to_zarr(f'{path_save}/dm_estimation.zarr', consolidated=True)
    print('done')


if __name__ == '__main__':
    main()
