#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from re import split
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from metpy.calc import pressure_to_height_std as p2h
from metpy.calc import lcl
from sqlalchemy.exc import OperationalError
import pandas as pd
from pytmatrix import tmatrix_aux, refractive, tmatrix, radar
from pymiecoated import Mie
from scipy.constants import c

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def comp_lwc(t, p):
    """
    """
    g = 9.8076
    cp = 1004.0
    Rd = 287.0
    Rv = 461.5
    # t = 273.15 + 8.0
    # p = 900.0
    L = 2.5e6
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (243.5 + (t - 273.15)))
    molar_mass_ratio = 1.0 / 1.608  # Rd/Rv
    r = molar_mass_ratio * es / p
    gamma_d = -g / cp
    gamma_m = gamma_d * (1.0 + L * r / (Rd * t)) / (1 + L ** 2 * r / (cp * Rv * t ** 2))
    rho_a = 100.0 * p / (Rd * t)  # hPa back to Pa for density calculation
    cw = rho_a * (cp / L) * (gamma_m - gamma_d)  # 1e3 for units. End result in g/m^3
    return cw


def bcksct(ds, instrument, ar=1, j=0) -> pd.DataFrame:
    """

    :param ds: numpy array of particle diameters. should be in millimeters
    :param ar: axis ratio of the particle
    :param j: Zenith angle input
    :return:
    """
    x_ku = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_Ku
    x_ka = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_Ka
    x_w = 2 * np.pi * (ds / 2.) / tmatrix_aux.wl_W
    # Tmatrix calculations
    tmat_ku = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_Ku,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ku], axis_ratio=1.0 / ar, thet0=j,
                                                   thet=180 - j,
                                                   phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
               i in ds]
    tmat_ka = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_Ka,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ka], axis_ratio=1.0 / ar, thet0=j,
                                                   thet=180 - j,
                                                   phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
               i in ds]
    tmat_w = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_W,
                                                  m=refractive.m_w_0C[tmatrix_aux.wl_W], axis_ratio=1.0 / ar, thet0=j,
                                                  thet=180 - j,
                                                  phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
              i in ds]

    # Mie calculations
    mie_ku = [Mie(x=x_ku[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ku]).qb() * np.pi * (i / 2.) ** 2 for w, i in
              enumerate(ds)]
    mie_ka = [Mie(x=x_ka[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ka]).qb() * np.pi * (i / 2.) ** 2 for w, i in
              enumerate(ds)]
    mie_w = [Mie(x=x_w[w], m=refractive.m_w_0C[tmatrix_aux.wl_W]).qb() * np.pi * (i / 2.) ** 2 for w, i in
             enumerate(ds)]
    df_scatter = pd.DataFrame(
        {'T_mat_Ku': tmat_ku, 'T_mat_Ka': tmat_ka, 'T_mat_W': tmat_w, 'Mie_Ku': mie_ku, 'Mie_Ka': mie_ka,
         'Mie_W': mie_w}, index=ds)
    path_db = f'{path_data}/db'
    str_db = f"sqlite:///{path_db}/backscatter.sqlite"
    df_scatter.to_sql(f'{instrument}', con=str_db, if_exists='replace')
    return df_scatter


def ref_calc(xr_data, mie=False):
    ds = xr_data.diameter.values / 1e3
    try:
        path_db = f'{path_data}/db'
        make_dir(path_db)
        str_db = f"sqlite:///{path_db}/backscatter.sqlite"
        backscatter = pd.read_sql(f"{xr_data.attrs['instrument']}", con=str_db)
    except OperationalError:
        backscatter = bcksct(ds, xr_data.attrs['instrument'])

    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    bcks = xr.Dataset.from_dataframe(backscatter).rename_dims({'index': 'diameter'}).rename({'index': 'diameter'})

    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.Mie_Ku * xr_data.psd * 1e6 * xr_data.d_d
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.Mie_Ka * xr_data.psd * 1e6 * xr_data.d_d
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.Mie_W * xr_data.psd * 1e6 * xr_data.d_d
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.T_mat_Ku * xr_data.psd * 1e6 * xr_data.d_d / 1e3
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.T_mat_Ka * xr_data.psd * 1e6 * xr_data.d_d / 1e3
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * bcks.T_mat_W * xr_data.psd * 1e6 * xr_data.d_d / 1e3

    ref = z_ka.to_dataset(name='z_ka')
    ref['z_ku'] = z_ku
    ref['z_w'] = z_w
    return ref


def pds_parameters(xr_data):
    """
    Compute the psd parameters
    :param xr_data: partice size distribution in # L-1 um-1
    :return: list with lwc, dm, nw, z, and r
    """
    lwc = (np.pi / (6 * 1000.)) * (xr_data.psd * 1e6) * (xr_data.diameter * 1e-3) ** 3 * (xr_data.d_d * 1e-3)
    m4 = (xr_data.psd * 1e6) * (xr_data.diameter * 1e-3) ** 4 * xr_data.d_d * 1e-3
    m3 = (xr_data.psd * 1e6) * (xr_data.diameter * 1e-3) ** 3 * xr_data.d_d * 1e-3
    dm = m4.sum('diameter') / m3.sum('diameter')
    z = xr_data.psd * 1e6 * (xr_data.diameter * 1e-3) ** 6 * xr_data.d_d
    nw = 1e3 * (4 ** 4 / np.pi) * (lwc.sum('diameter') / dm ** 4)
    params = lwc.to_dataset(name='lwc')
    params['dm'] = dm
    params['z'] = z
    params['nw'] = nw
    return params


def main():
    aircraft = 'Lear'
    aircraft2 = 'Learjet'
    ds = xr.open_dataset('C:/Users/alfonso8/Downloads/CAMP2EX_AVAPS_RD41_v1_20190907_030551.nc')
    # ds = xr.open_dataset('C:/Users/alfonso8/Downloads/CAMP2EX_AVAPS_RD41_v1_20190907_032811.nc')
    store3 = f"{path_data}/zarr/2DS10_{aircraft2}.zarr"
    xr_2ds = xr.open_zarr(store3).sel(time=slice('2019-09-07 02:31:30', '2019-09-07 02:34:10'))

    pds_parameters(xr_2ds)
    p = ds.dropna('time')['pres']
    t = ds.dropna('time')['tdry']
    dp = ds.dropna('time')['dp']
    lcl_p, lcl_t = lcl(p, t, dp)
    cbh = p2h(lcl_p[0])
    ds = ds.where(ds.pres < lcl_p[0].m)
    p = ds['pres'].dropna('time')
    t = ds['tdry'].dropna('time')
    dp = ds['dp'].dropna('time')
    h = p2h(p)
    lwc = comp_lwc(t.values + 273.15, p.values) #, h.values, cbh=cbh.m)
    lwc_sum = np.cumsum(lwc[:-1] * h.diff('time').values * 1e6)
    fig, ax = plt.subplots()
    ax.plot(lwc_sum, h[:-1] * 1000)
    plt.show()
    pass


if __name__ == '__main__':
    main()
    pass
