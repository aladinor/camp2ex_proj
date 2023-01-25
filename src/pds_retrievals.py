#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.constants import c
from scipy.special import gamma
from scipy.optimize import minimize, brentq
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
    return ka - ku


def equ_fucnt(dm, xr_comb, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    dfr = xr_comb.dbz_t_ka - xr_comb.dbz_t_ku
    return dfr - (ka - ku)


def dm_solver(xr_comb, mu=3):
    dm_bounds = [[0.1, 3.5]]
    dm_const = {'type': 'eq', 'fun': equ_fucnt, 'args': (xr_comb, mu)}
    dm_0 = np.array([1.75])
    res = minimize(fun=objective_func, x0=dm_0, args=(xr_comb, mu), bounds=dm_bounds, constraints=dm_const, tol=1e-2,
                   options={'maxiter': 50})
    return xr.DataArray(res['x'], dims=['time'], coords=dict(time=(['time'], np.array([xr_comb.time.values]))))


def dm_optimization(dm1, dm2, xr_comb, mu=3, niter=100, tol=0.000001):
    # for i in range(niter):
    res = 1
    while res >= tol:
        fx1 = equ_fucnt(dm1, xr_comb=xr_comb, mu=mu).values
        fx2 = equ_fucnt(dm2, xr_comb=xr_comb, mu=mu).values
        f_diff = (fx2 - fx1) / (dm2 - dm1)
        dm_new = dm1 - (fx1 / f_diff)
        res = np.abs(dm_new-dm1)
        dm1 = dm_new
    return dm1


def rain_rate():
    rr = 6 * np.pi * 1e-4 * nw
    return


def main():
    xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_Lear_300_1000_4_bins.zarr')
    xr_comb = xr_comb.isel(time=range(15, 25))
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    # w_wvl = c / 95e9 * 1000
    #
    # ib_ku_1 = integral(dm=xr_comb.isel(time=20).dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3,
    #                    mu=xr_comb.isel(time=20).mu, band="Ku")
    # sol = [dm_optimization(dm1=1.5, dm2=2, xr_comb=i, mu=i.mu.values)
    #        for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]
    sol = dm_optimization(dm1=1.5, dm2=2, xr_comb=xr_comb.isel(time=5), mu=xr_comb.isel(time=0).mu.values)
    # Solve for Dm using DFR
    dfr = xr_comb.dbz_t_ka - xr_comb.dbz_t_ku
    print(1)
    dm_sol = [brentq(equ_fucnt, 0.5, 10, args=(i, 3))
                        for _, i in xr_comb.chunk(chunks={"time": 1}).groupby("time")]
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
