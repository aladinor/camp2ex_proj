#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import xarray as xr
from shutil import rmtree
from scipy.constants import c
from scipy.special import gamma
from zarr.errors import ContainsGroupError
from re import split
from pytmatrix import tmatrix_aux
from typing import Callable
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


def dfr_root(dm, d, d_d, dfr, mu=3, dataset='camp2ex'):
    ku_wvl = tmatrix_aux.wl_Ku
    ka_wvl = tmatrix_aux.wl_Ka
    if mu is None:
        if dataset == 'camp2ex':
            mu = 5.95 * dm ** 0.468 - 4
        else:
            mu = 11.1 * dm ** -0.72 - 4
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return dfr - ku + ka


def dfr_gamma(dm, d, d_d, mu=3):
    ku_wvl = tmatrix_aux.wl_Ku
    ka_wvl = tmatrix_aux.wl_Ka
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return ku - ka


def wrapper(dms, dm, ds):
    return xr.apply_ufunc(
        root,
        dms,
        dm,
        ds.dfr,
        ds.dbz_t_ku,
        input_core_dims=[['dm'], ['dm'], [], []],
        vectorize=True,
        dask='Parallelized'
    )


def root(dms_dfr, dm, dfr, z_ku):
    try:
        if (dfr < 0) & (z_ku < 10):
            idxmax = np.argmax(dms_dfr)
            dms = dms_dfr[:idxmax]
            idmin = np.argmin(np.abs(dms - 0.0))
            return dm[idmin]
        elif (dfr < 0) & (z_ku > 20):
            idxmax = np.argmax(dms_dfr)
            dms = dms_dfr[idxmax:]
            idmin = np.argmin(np.abs(dms - 0.0))
            return dm[idmin + idxmax]
        else:
            idmin = np.argmin(np.abs(dms_dfr - 0.0))
            return dm[idmin]
    except ValueError:
        return np.nan


def ib_cal(dm, mu, d, d_d, wv='Ku'):
    ku_wvl = tmatrix_aux.wl_Ku
    ka_wvl = tmatrix_aux.wl_Ka
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    if wv == 'Ku':
        return 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    else:
        return 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))


def nw_retrieval(z, dm, mu, d, d_d):
    return z - ib_cal(dm=dm, d=d, d_d=d_d, mu=mu)


def rain_retrieval(nw, mu, dm, d, d_d, vel_m='lerm'):
    lerm_vel: Callable[[float], float] = lambda diam: 9.25 * (1 - np.exp(-0.068 * diam ** 2 - 0.488 * diam))  # d in mm
    ulbr_vel: Callable[[float], float] = lambda diam: 3.78 * diam ** 0.67  # with d in mm
    if vel_m == 'lemr':
        vel = lerm_vel(d)
    else:
        vel = ulbr_vel(d)
    f_mu = (6 * (mu + 4) ** (mu + 4)) / (4 ** 4 * gamma(mu + 4))
    r = 6 * np.pi * 1e-4 * (nw * f_mu * (d / dm) ** mu * np.exp(-(4 + mu) * (d / dm)) * vel * d ** 3 * d_d)
    return r.sum('diameter')


def dm_retrieval(ds):
    dm = np.arange(0.01, 4, 0.001)
    mus = np.tile(3, ds.dfr.shape[0])
    mus = xr.DataArray(data=mus,
                       dims=['time'],
                       coords=dict(time=(['time'], ds.time.values)))
    dms = xr.DataArray(data=dm,
                       dims=['dm'],
                       coords=dict(dm=(['dm'], dm)))

    # dm - DFR (N(D), sigma_b) -  True values
    dms_dfr = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, mu=ds.mu, dfr=ds.dfr)
    ds_sol = wrapper(dms=dms_dfr.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())
    ds_sol = ds_sol.to_dataset(name='dm_rt_dfr_nd')
    #
    # # dm - DFR(mu, dm) gamma-shaped  -  True values
    # dfr_gm = dfr_gamma(dm=ds.dm, d=ds.diameter, d_d=ds.d_d, mu=ds.mu)
    # dms_dfr_gamma = dfr_root(dm=dms, d=ds.diameter, d_d=ds.d_d, mu=ds.mu, dfr=dfr_gm)
    # ds_sol['dm_rt_dfr_gm'] = wrapper(dms=dms_dfr_gamma.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())
    #
    # # dm - DFR    # dm - DFR(mu=3,dm, nw)
    # dms_dfr_mu_3 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=dfr_gm, mu=mus)
    # ds_sol['dm_rt_dfr_gm_mu_3'] = wrapper(dms=dms_dfr_mu_3.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())
    #
    # # dm - DFR    # dm - DFR(mu=3,dm, nw)
    # dms_dfr_mu_3 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=mus)
    # ds_sol['dm_rt_dfr_nd_mu_3'] = wrapper(dms=dms_dfr_mu_3.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())

    # dm - DFR using williams et al. 2014
    dms_will = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=None, dataset='williams')
    ds_sol['dm_williams'] = wrapper(dms_will.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())
    ds_sol['mu_williams'] = 11.1 * ds_sol.dm_williams ** -0.72 - 4

    # dm - DFR using camp2ex
    dms_camp = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=None)
    ds_sol['dm_camp'] = wrapper(dms_camp.load(), dm=dm, ds=ds[['dfr', 'dbz_t_ku']].load())
    ds_sol['mu_camp'] = 5.98 * ds_sol.dm_camp ** 0.468 - 4

    # Adding aditional information from
    ds_sol['dfr'] = (['time'], ds.dfr.values)
    ds_sol['dfr_gm'] = (['time'], dfr_gm.values)
    ds_sol['dm_true'] = (['time'], ds.dm.values)

    log10nw_true = nw_retrieval(z=ds.dbz_t_ku, dm=ds.dm, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_gm = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_rt_dfr_gm, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_nd = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_rt_dfr_nd, mu=mus, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_gm_mu_3 = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_rt_dfr_gm_mu_3, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_nd_mu_3 = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_rt_dfr_nd_mu_3, mu=mus, d=ds.diameter, d_d=ds.d_d)
    log10nw_will = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_williams, mu=ds_sol.mu_williams, d=ds.diameter, d_d=ds.d_d)
    log10nw_camp = nw_retrieval(z=ds.dbz_t_ku, dm=ds_sol.dm_camp, mu=ds_sol.mu_camp, d=ds.diameter, d_d=ds.d_d)

    ds_sol['log10nw_true'] = (['time'], 10 * ds.log10_nw.values)
    ds_sol['log10nw_true_mu_dm'] = (['time'], log10nw_true.values)
    ds_sol['log10nw_dm_gm'] = (['time'], log10nw_dm_gm.values)
    ds_sol['log10nw_dm_nd'] = (['time'], log10nw_dm_nd.values)
    ds_sol['log10nw_dm_gm_mu_3'] = (['time'], log10nw_dm_gm_mu_3.values)
    ds_sol['log10nw_dm_nd_mu_3'] = (['time'], log10nw_dm_nd_mu_3.values)
    ds_sol['log10nw_will'] = (['time'], log10nw_will.values)
    ds_sol['log10nw_camp'] = (['time'], log10nw_camp.values)

    rain_true = rain_retrieval(nw=10 ** (log10nw_true / 10),
                               mu=ds.mu, dm=ds.dm,
                               d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_gm = rain_retrieval(nw=10 ** (log10nw_dm_gm / 10),
                                mu=ds.mu, dm=ds_sol.dm_rt_dfr_gm,
                                d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_nd = rain_retrieval(nw=10 ** (log10nw_dm_nd / 10),
                                mu=mus, dm=ds_sol.dm_rt_dfr_nd,
                                d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_gm_mu_3 = rain_retrieval(nw=10 ** (log10nw_dm_gm_mu_3 / 10),
                                     mu=ds.mu, dm=ds_sol.dm_rt_dfr_gm_mu_3,
                                     d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_nd_mu_3 = rain_retrieval(nw=10 ** (log10nw_dm_nd_mu_3 / 10),
                                     mu=mus, dm=ds_sol.dm_rt_dfr_nd_mu_3,
                                     d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_will = rain_retrieval(nw=10 ** (log10nw_will / 10), mu=ds_sol.mu_williams, dm=ds_sol.dm_williams,
                               d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_camp = rain_retrieval(nw=10 ** (log10nw_camp / 10), mu=ds_sol.mu_camp, dm=ds_sol.dm_camp,
                               d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    ds_sol['r_true'] = (['time'], ds.r.values)
    ds_sol['r_true_nw_mu_dm'] = (['time'], rain_true.values)
    ds_sol['r_dm_gm'] = (['time'], rain_dm_gm.values)
    ds_sol['r_dm_nd'] = (['time'], rain_dm_nd.values)
    ds_sol['r_dm_gm_mu_3'] = (['time'], rain_dm_gm_mu_3.values)
    ds_sol['r_gpm'] = (['time'], rain_dm_nd_mu_3.values)
    ds_sol['r_will'] = (['time'], rain_will.values)
    ds_sol['r_camp'] = (['time'], rain_camp.values)
    ds_sol['r_gpm_operational'] = 1.370 * 1 ** 4.258 * ds_sol.dm_rt_dfr_gm_mu_3 ** 5.420
    return ds_sol


def main():
    for i in ['Lear', 'P3B']:
        xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_{i}_600_1000_5_bins_merged.zarr')
        dm = dm_retrieval(xr_comb)
        save_path = f'{path_data}/cloud_probes/zarr/dm_retrieved_{i}_corr.zarr'
        try:
            _ = dm.to_zarr(save_path, consolidated=True)
        except ContainsGroupError:
            rmtree(save_path)
            _ = dm.to_zarr(save_path, consolidated=True)

        xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_{i}_600_1000_5_bins_merged_5s.zarr')
        dm = dm_retrieval(xr_comb)
        save_path = f'{path_data}/cloud_probes/zarr/dm_retrieved_{i}_corr_5s.zarr'
        try:
            _ = dm.to_zarr(save_path, consolidated=True)
        except ContainsGroupError:
            rmtree(save_path)
            _ = dm.to_zarr(save_path, consolidated=True)
        print('Done!!!')


if __name__ == '__main__':
    main()
