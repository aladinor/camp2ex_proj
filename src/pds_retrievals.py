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


def eq_funct(dm, xr_comb, dfr, mu=3):
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    ib_ku = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    ib_ka = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ka")
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
    dms = xr.DataArray(data=dm,
                       dims=['dm'],
                       coords=dict(dm=(['dm'], dm)))
    # dm - DFR (N(D), sigma_b)
    rest = eq_funct(dms, ds, mu=ds.mu, dfr=ds.dfr)
    rest = rest.to_dataset(name='dms_dfr')
    dm_idx = dm_filt(rest.load())
    rest['dm_rt_dfr'] = (['time'], rest.isel(dm=dm_idx.dms_dfr).dm.values)

    # dm - DFR(mu, dm)
    dfr = dfr_norm(dm=ds.dm, d=ds.diameter,d_d=ds.d_d, mu=ds.mu)
    rest2 = eq_funct(dms, ds, mu=ds.mu, dfr=dfr)
    rest['dms_norm_dfr'] = (["time", 'dm'], rest2.values)
    dm_idx2 = dm_filt(rest2.load())
    rest['dm_rt_norm_dfr'] = (['time'], rest2.isel(dm=dm_idx2).dm.values)
    rest['dfr'] = (['time'], ds.dfr.values)
    rest['dfr_mudm'] = (['time'], dfr.values)
    rest['dm_true'] = (['time'], ds.dm.values)
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
