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
    # return np.unravel_index(np.argmin(np.abs(x - value)), x.shape)[0]


def dfr_root(dm, d, d_d, dfr, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=d / 1e3, dd=d_d / 1e3, mu=mu, band="Ka")
    ku = 10 * np.log10(((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    ka = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka))
    return dfr - ku + ka


def dfr_norm(dm, d, d_d, mu=3):
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
    )


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
    dm_idx = dm_filt(rest.load())
    rest['dm_rt_dfr'] = (['time'], rest.isel(dm=dm_idx.dms_dfr).dm.values)

    # dm - DFR(mu, dm)
    dfr = dfr_norm(dm=ds.dm, d=ds.diameter, d_d=ds.d_d, mu=ds.mu)
    rest2 = dfr_root(dm=dms, d=ds.diameter, d_d=ds.d_d, mu=ds.mu, dfr=dfr)
    rest['dms_norm_dfr'] = (["time", 'dm'], rest2.values)
    dm_idx2 = dm_filt(rest2.load())
    rest['dm_rt_norm_dfr'] = (['time'], rest2.isel(dm=dm_idx2).dm.values)
    rest['dfr'] = (['time'], ds.dfr.values)
    rest['dfr_mudm'] = (['time'], dfr.values)
    rest['dm_true'] = (['time'], ds.dm.values)

    # dm - DFR    # dm - DFR(mu=3, N(D), sigma_b)
    rest3 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=ds.dfr, mu=mus)
    rest['dms_mu_3'] = (["time", 'dm'], rest3.values)
    dm_idx3 = dm_filt(rest3.load())
    rest['dm_rt_dfr_mu_3'] = (['time'], rest3.isel(dm=dm_idx3).dm.values)

    # dm - DFR    # dm - DFR(mu=3,dm, nw)
    dfr_mu = dfr_norm(dm=ds.dm, d=ds.diameter, d_d=ds.d_d, mu=ds.mu)
    rest4 = dfr_root(dms, d=ds.diameter, d_d=ds.d_d, dfr=dfr_mu, mu=mus)
    rest['dms_mu_3'] = (["time", 'dm'], rest4.values)
    dm_idx4 = dm_filt(rest4.load())
    rest['dm_rt_norm_dfr_mu_3'] = (['time'], rest4.isel(dm=dm_idx4).dm.values)
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


if __name__ == '__main__':
    main()
