#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
from sqlalchemy.exc import OperationalError
import xarray as xr
from scipy.constants import c
from scipy.special import gamma
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


def main():
    xr_comb = xr.open_zarr(f'{path_data}/cloud_probes/zarr/combined_psd_Lear_300_1000_4_bins.zarr')
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    t = 20
    dm = xr_comb.isel(time=t).dm.values
    mu = xr_comb.isel(time=t).mu.values
    nw = xr_comb.isel(time=t).nw.values
    z_ku = xr_comb.isel(time=t).dbz_t_ku.values
    dfr = xr_comb.dbz_t_ka - xr_comb.dbz_t_ku
    ib_ku_1 = integral(dm=dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=mu, band="Ku")
    z = 10 * np.log10(nw * (ku_wvl ** 4 / (np.pi ** 5 * 0.93) * ib_ku_1)).values
    ib_ku = integral(dm=xr_comb.dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=xr_comb.mu, band="Ku")
    ib_ku_const = integral(dm=xr_comb.dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=3, band="Ku")
    ib_ka = integral(dm=xr_comb.dm, d=xr_comb.diameter / 1e3, dd=xr_comb.d_d / 1e3, mu=xr_comb.mu, band="Ka")
    dfr_cal = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka) /
                            ((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku))
    dfr_gpm = 10 * np.log10(((ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ka) /
                            ((ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ib_ku_const))
    z_xr = 10 * np.log10(xr_comb.nw * (ku_wvl ** 4 / (np.pi ** 5 * 0.93) * ib_ku))

    fig, ax = plt.subplots()
    ax.scatter(dfr, dfr_cal)
    ax.scatter(dfr, dfr_gpm)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.show()
    print(1)
    pass


if __name__ == '__main__':
    main()
