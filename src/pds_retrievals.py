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
from typing import Callable
from numba import float64, guvectorize, int64, jit
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


def find_nearest(x, value=0.0):
    return np.argmin(np.abs(x - value))


def dfr_root(dm, d, d_d, dfr, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return dfr - ku + ka


def dfr_gamma(dm, d, d_d, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return ku - ka


def dm_filt(da, dim='dm'):
    return xr.apply_ufunc(
        find_nearest,
        da,
        input_core_dims=[[dim]],
        kwargs=dict(value=0.),
        vectorize=True,
        dask='Parallelized'
    )


def ib_cal(dm, mu, d, d_d, wv='Ku'):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
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
    dm = np.arange(0.1, 4, 0.001)
    mus = np.tile(3, ds.dfr.shape[0])
    mus = xr.DataArray(data=mus,
                       dims=['time'],
                       coords=dict(time=(['time'], ds.time.values)))
    dms = xr.DataArray(data=dm,
                       dims=['dm'],
                       coords=dict(dm=(['dm'], dm)))

    # dm - DFR (N(D), sigma_b)
    rest = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, mu=ds.mu, dfr=ds.dfr)
    rest = rest.to_dataset(name='dms_dfr')
    # dm_idx = dm_filt(rest.load())
    # rest['dm_rt_dfr'] = (['time'], rest.isel(dm=dm_idx.dms_dfr).dm.values)
    dm_idx = np.argmin(np.abs(rest.dms_dfr.data - 0.0), axis=1)
    rest['dm_rt_dfr'] = (['time'], rest.isel(dm=dm_idx).dm.values)

    # dm - DFR(mu, dm)
    dfr_gm = dfr_gamma(dm=ds.dm, d=ds.diameter, d_d=ds.d_d, mu=ds.mu)
    rest2 = dfr_root(dm=dms, d=ds.diameter, d_d=ds.d_d, mu=ds.mu, dfr=dfr_gm)
    rest['dms_dfr_gm'] = (["time", 'dm'], rest2.values)
    # dm_idx2 = dm_filt(rest2.load())
    dm_idx2 = np.argmin(np.abs(rest2.data - 0.0), axis=1)
    rest['dm_rt_dfr_gm'] = (['time'], rest2.isel(dm=dm_idx2).dm.values)
    rest['dfr'] = (['time'], ds.dfr.values)
    rest['dfr_gm'] = (['time'], dfr_gm.values)
    rest['dm_true'] = (['time'], ds.dm.values)

    # dm - DFR    # dm - DFR(N(D), sigma_b)
    rest3 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=ds.mu)
    rest['dms_dfr_nd'] = (["time", 'dm'], rest3.values)
    # dm_idx3 = dm_filt(rest3.load())
    dm_idx3 = np.argmin(np.abs(rest3.data - 0.0), axis=1)
    rest['dm_rt_dfr_nd'] = (['time'], rest3.isel(dm=dm_idx3).dm.values)

    # dm - DFR    # dm - DFR(mu=3,dm, nw)
    rest4 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=dfr_gm, mu=mus)
    rest['dms_dfr_gm_mu_3'] = (["time", 'dm'], rest4.values)
    # dm_idx4 = dm_filt(rest4.load())
    dm_idx4 = np.argmin(np.abs(rest4.data - 0.0), axis=1)
    rest['dm_rt_dfr_gm_mu_3'] = (['time'], rest4.isel(dm=dm_idx4).dm.values)

    # dm - DFR    # dm - DFR(mu=3,dm, nw)
    rest5 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=mus)
    rest['dms_dfr_nd_mu_3'] = (["time", 'dm'], rest5.values)
    # dm_idx5 = dm_filt(rest5.load())
    dm_idx5 = np.argmin(np.abs(rest5.data - 0.0), axis=1)
    rest['dm_rt_dfr_nd_mu_3'] = (['time'], rest5.isel(dm=dm_idx5).dm.values)

    log10nw_true = nw_retrieval(z=ds.dbz_t_ku, dm=ds.dm, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_gm = nw_retrieval(z=ds.dbz_t_ku, dm=rest.dm_rt_dfr_gm, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_nd = nw_retrieval(z=ds.dbz_t_ku, dm=rest.dm_rt_dfr_nd, mu=mus, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_gm_mu_3 = nw_retrieval(z=ds.dbz_t_ku, dm=rest.dm_rt_dfr_gm_mu_3, mu=ds.mu, d=ds.diameter, d_d=ds.d_d)
    log10nw_dm_nd_mu_3 = nw_retrieval(z=ds.dbz_t_ku, dm=rest.dm_rt_dfr_nd_mu_3, mu=mus, d=ds.diameter, d_d=ds.d_d)

    rest['log10nw_true'] = (['time'], log10nw_true.values)
    rest['log10nw_dm_gm'] = (['time'], log10nw_dm_gm.values)
    rest['log10nw_dm_nd'] = (['time'], log10nw_dm_nd.values)
    rest['log10nw_dm_gm_mu_3'] = (['time'], log10nw_dm_gm_mu_3.values)
    rest['log10nw_dm_nd_mu_3'] = (['time'], log10nw_dm_nd_mu_3.values)

    rain_true = rain_retrieval(nw=10 ** (log10nw_true / 10),
                               mu=ds.mu, dm=ds.dm,
                               d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_gm = rain_retrieval(nw=10 ** (log10nw_dm_gm / 10),
                                mu=ds.mu, dm=rest.dm_rt_dfr_gm,
                                d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_nd = rain_retrieval(nw=10 ** (log10nw_dm_nd / 10),
                                mu=mus, dm=rest.dm_rt_dfr_nd,
                                d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_gm_mu_3 = rain_retrieval(nw=10 ** (log10nw_dm_gm_mu_3 / 10),
                                     mu=ds.mu, dm=rest.dm_rt_dfr_gm_mu_3,
                                     d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rain_dm_nd_mu_3 = rain_retrieval(nw=10 ** (log10nw_dm_nd_mu_3 / 10),
                                     mu=mus, dm=rest.dm_rt_dfr_nd_mu_3,
                                     d=ds.diameter / 1e3, d_d=ds.d_d / 1e3)

    rest['r_true'] = (['time'], rain_true.values)
    rest['r_dm_gm'] = (['time'], rain_dm_gm.values)
    rest['r_dm_nd'] = (['time'], rain_dm_nd.values)
    rest['r_dm_gm_mu_3'] = (['time'], rain_dm_gm_mu_3.values)
    rest['r_gpm'] = (['time'], rain_dm_nd_mu_3.values)
    return rest


def main():
    for i in ['Lear', 'P3B']:
        xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_{i}_600_1000_5_bins_merged.zarr')
        dm = dm_retrieval(xr_comb)
        save_path = f'{path_data}/cloud_probes/zarr/dm_retrieved_{i}.zarr'
        try:
            _ = dm.to_zarr(save_path, consolidated=True)
        except ContainsGroupError:
            rmtree(save_path)
            _ = dm.to_zarr(save_path, consolidated=True)
        print('Done!!!')
        # resampling to 5s
        # xr_res = xr_comb.resample(time='5S').mean()
        # dm = dm_retrieval(xr_res)
        # save_path = f'{path_data}/cloud_probes/zarr/dm_retrieved_{i}_5s_res.zarr'
        # try:
        #     _ = dm.to_zarr(save_path, consolidated=True)
        # except ContainsGroupError:
        #     rmtree(save_path)
        #     _ = dm.to_zarr(save_path, consolidated=True)


if __name__ == '__main__':
    main()
