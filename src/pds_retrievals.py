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


def equ_fucnt(dm, xr_comb, mu=3, dfr=None):
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
    dm_const = {'type': 'eq', 'fun': equ_fucnt, 'args': (xr_comb, mu)}
    dm_0 = np.array([1.75])
    res = minimize(fun=objective_func, x0=dm_0, args=(xr_comb, mu), bounds=dm_bounds, constraints=dm_const, tol=1e-2,
                   options={'maxiter': 50})
    return xr.DataArray(res['x'], dims=['time'], coords=dict(time=(['time'], np.array([xr_comb.time.values]))))


def dm_retrieval(dm1, dm2, xr_comb, mu=3, tol=0.00000001, maxiter=100):
    res = 1
    _iter = 0
    ku_ka = objective_func(xr_comb.dm, xr_comb, xr_comb.mu).values
    while not (_iter > maxiter or tol > res):
        fx1 = equ_fucnt(dm1, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        fx2 = equ_fucnt(dm2, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        dm_new = dm1 - fx1 * (dm2 - dm1) / (fx2 - fx1)
        res = np.abs(dm_new-dm2)
        if np.isnan(res):
            return np.nan
        dm2 =dm1
        dm1 = dm_new
        _iter += 1
        print(_iter)
    print(f'converged: {dm1}')
    return dm1


def dm_plt(dm1, dm2, xr_comb, mu=3, tol=0.00000001, maxiter=100):
    res = 1
    _iter = 0
    dm_s = []
    f_res = []
    f_res2 = []
    x = np.arange(0.1, 5, 0.1)
    ku_ka = objective_func(xr_comb.dm, xr_comb, xr_comb.mu).values
    dm_plot = np.array([equ_fucnt(i, xr_comb=xr_comb, mu=mu, dfr=ku_ka) for i in x])
    dm_plot2 = np.array([equ_fucnt(i, xr_comb=xr_comb, mu=mu) for i in x])
    dfr = (xr_comb.dbz_t_ku - xr_comb.dbz_t_ka).values

    dfr_ib = dfr - ku_ka
    fig, ax = plt.subplots(dpi=180)
    ax.plot(x, dm_plot, label='est')
    ax.plot(x, dm_plot2, label='real')
    ax.scatter(xr_comb.dm, dfr_ib)
    ax.legend()
    while not (_iter > maxiter or tol > res):
        fx1 = equ_fucnt(dm1, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        fx2 = equ_fucnt(dm2, xr_comb=xr_comb, mu=mu, dfr=ku_ka).values
        dm_new = dm1 - fx1 * (dm2 - dm1) / (fx2 - fx1)
        res = np.abs(dm_new-dm2)
        if np.isnan(res):
            return np.nan
        dm2 = dm1
        dm1 = dm_new
        dm_s.append(dm1)
        f_res.append(fx1)
        f_res2.append(fx2)
        _iter += 1
        ax.scatter(dm_new, fx1, c='b', s=0.8, label='fx1')
        ax.scatter(dm1, fx2, c='k', s=0.8, label='fx2')
        ax.set_xlabel('Dm')
        ax.set_ylabel(f'DFR ({(xr_comb.dbz_t_ku - xr_comb.dbz_t_ka).values:.2f}) - Ib-Ku + Ib-Ka')
        # plt.show()
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

    # sol = dm_retrieval(dm1=0.5, dm2=4., xr_comb=xr_comb.isel(time=1), mu=xr_comb.isel(time=1).mu.values)
    # sol2 = brentq(objective_func, 0.5, 4., args=(xr_comb.isel(time=1), xr_comb.isel(time=1).mu.values))
    dm_real = xr_comb.isel(time=range(3)).dm.values
    # see convergence in plot
    # test_dm = dm_plt(dm1=0.5, dm2=3.5, xr_comb=xr_comb.isel(time=6), mu=xr_comb.isel(time=6).mu.values)

    # multiple test
    dm_sol = [dm_retrieval(dm1=0.5, dm2=3.5, xr_comb=i.load(), mu=i.mu.load())
              for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]

    dm_sol_1 = [brentq(equ_fucnt, 0.5, 3, args=(i, 3))
                        for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]
    # Solve for Dm using DFR
    dm_real = xr_comb.dm.values
    dfr = xr_comb.dbz_t_ka - xr_comb.dbz_t_ku

    dm_sol = xr.concat([dm_solver(i, mu=3) for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")], 'time')

    fig, ax = plt.subplots()
    sc = ax.scatter(xr_comb.dm, dm_sol, c=xr_comb.r)
    ax.set_ylabel('Dm GPM')
    ax.set_xlabel('Dm Truth')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.colorbar(sc, label='R (mmhr-1)')
    plt.savefig('../results/dm_est.png')
    plt.show()
    print(1)

    # ib_ku_1 = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    # z = 10 * np.log10(nw * (ku_wvl ** 4 / (np.pi ** 5 * 0.93) * ib_ku_1)).values

    ib_ku_gpm = integral(dm=dm_sol, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=3, band="Ku")
    ib_ka_gpm = integral(dm=dm_sol, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=xr_comb.mu, band="Ka")

    dfr_gpm = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka_gpm) /
                            ((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku_gpm))

    fig, ax = plt.subplots()
    sc = ax.scatter(dfr, dfr_gpm, c=xr_comb.r)
    ax.set_ylabel('DFR GPM')
    ax.set_xlabel('DFR Measured')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.colorbar(sc, label='R (mmhr-1)')
    plt.savefig('../results/dm_gpm.png')
    plt.show()
    print(1)

    ib_ku = integral(dm=xr_comb.dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=xr_comb.mu, band="Ku")
    ib_ka = integral(dm=xr_comb.dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=xr_comb.mu, band="Ka")
    dfr_cal = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka) /
                            ((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    fig, ax = plt.subplots()
    sc = ax.scatter(dfr, dfr_cal, c=xr_comb.r)
    ax.set_ylabel('DFR Calculated')
    ax.set_xlabel('DFR Measured')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.colorbar(sc, label='R (mmhr-1)')
    plt.savefig('../results/dfr_est.png')
    plt.show()
    print(1)

    log10_nw_GPM = xr_comb.dbz_t_ku - 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku_gpm))
    fig, ax = plt.subplots()
    sc = ax.scatter(xr_comb.log10_nw, log10_nw_GPM/10, c=xr_comb.r)
    ax.set_ylabel('log10(Nw) GPM')
    ax.set_xlabel('log10(Nw) Truth')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.colorbar(sc, label='R (mmhr-1)')
    plt.savefig('../results/nw_gpm.png')
    plt.show()
    print(1)
    pass


if __name__ == '__main__':
    main()
