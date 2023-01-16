#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
from typing import Callable
import numpy as np
import pandas as pd
from sqlalchemy.exc import OperationalError
from pytmatrix import tmatrix_aux, refractive, tmatrix, radar
from pytmatrix.scatter import ext_xsect
from pymiecoated import Mie
from scipy.constants import c
from re import split

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def get_data(aircraft='Lear', sensors=None, temp=2):
    """
    Functions that retrieves cloud probe data from pkl archives
    :param sensors: sensor which data will be retrieved
    :param aircraft: aircraft
    :param temp: temperature for filtering
    :return: list of available dataframe in CAMP2ex
    """
    if not sensors:
        sensors = ['FCDP', '2DS10', 'HVPS']
    if aircraft == 'Lear':
        ls_lear = [glob.glob(f'{path_data}/cloud_probes/pkl/{i}*_Learjet.pkl')[0] for i in sensors]
        if os.name == 'nt':
            ls_lear = sorted([i for i in ls_lear])
        lear_df = [pd.read_pickle(i) for i in ls_lear]
        _attrs = [i.attrs for i in lear_df]
        if temp:
            try:
                lear_df = [i[i['Temp'] > temp] for i in lear_df]
            except KeyError:
                ls_temp = glob.glob(f'{path_data}/cloud_probes/pkl/Page0*.pkl')[0]
                lear_df = [pd.merge(i, pd.read_pickle(ls_temp)['Temp'], right_index=True, left_index=True)
                           for i in lear_df]
                for i, attrs in enumerate(_attrs):
                    lear_df[i].attrs = attrs
        return lear_df
    elif aircraft == 'P3B':
        ls_p3 = sorted([glob.glob(f'{path_data}/cloud_probes/pkl/{i}*_P3B.pkl')[0] for i in sensors])
        p3_df = [pd.read_pickle(i) for i in ls_p3]
        _attrs = [i.attrs for i in p3_df]
        if temp:
            try:
                p3_df = [i[i['Temp'] > temp] for i in p3_df]
            except KeyError:
                p3_temp = pd.read_pickle(glob.glob(f'{path_data}/cloud_probes/pkl/p3b_merge.pkl')[0])
                p3_df = [pd.merge(i, p3_temp[' Static_Air_Temp_YANG_MetNav'], left_index=True, right_index=True)
                         for i in p3_df]
                temp = 2
                for i, df in enumerate(p3_df):
                    df.attrs = _attrs[i]
                    df.rename(columns={' Static_Air_Temp_YANG_MetNav': 'Temp'}, inplace=True)
                    df = df[df['Temp'] >= temp]
                    p3_df[i] = df
        return p3_df
    else:
        raise TypeError(f"{aircraft} not available. Use Lear or P3B")


def _scatterer(diameters, ar, wl, j=0, rt=tmatrix.Scatterer.RADIUS_MAXIMUM, forward=True):
    """
    Function that computes the scatterer object using pytmatrix package
    :param diameters: numpy array with diameters in mm
    :param ar: numpy array with axis ratio values
    :param wl: wavelength for which scatterer will be computed. e.g "Ka"
    :param j: Zenith angle input
    :param rt: maximum radius in mm
    :param forward: True if forward scattering geometry. False will use backward scattering geometry.
    :return: list of scatterers objects
    """
    if forward:
        gm = tmatrix_aux.geom_horiz_forw
    else:
        gm = tmatrix_aux.geom_horiz_back

    if wl == "Ku":
        wlg = tmatrix_aux.wl_Ku
        m = refractive.m_w_0C[tmatrix_aux.wl_Ku]
    elif wl == "Ka":
        wlg = tmatrix_aux.wl_Ka
        m = refractive.m_w_0C[tmatrix_aux.wl_Ka]
    elif wl == "W":
        wlg = tmatrix_aux.wl_W
        m = refractive.m_w_0C[tmatrix_aux.wl_W]
    else:
        raise Exception('wl {} not valid. Please use Ka, Ku, or W'.format(wl))

    return [tmatrix.Scatterer(radius=d / 2., wavelength=wlg, m=m, axis_ratio=1.0 / ar[idx], thet0=j,
                              thet=180 - j, phi0=0., phi=180., radius_type=rt,
                              geometry=gm) for idx, d in enumerate(diameters)]


def bck_extc_crss(diameters, instrument, _lower=300, _upper=1000, ar=None, j=0) -> pd.DataFrame:
    """
    Function that computes the backscatter and extinction cross-section for a particle.
    :param diameters: numpy array of diameters (mm) to which bcksct or extc will be computed.
    :param instrument: instrument for naming the database where bcksct or extc will be stored. e.g. "2DS"
    :param _lower: min diameter.
    :param _upper: max diameter
    :param ar: axis ratio numpy array
    :param j: Zenith angle input
    :return: Pandas dataframe with backscattering and extinction cross-section for Ku, Ka, and W band radar
    """

    try:
        path_db = f'{path_data}/cloud_probes/db'
        str_db = f"sqlite:///{path_db}/scattering_{_lower}_{_upper}.sqlite"
        df_scatter = pd.read_sql(f"{instrument}", con=str_db).set_index('index')
        return df_scatter
    except (OperationalError, ValueError):
        if not ar:
            andsager_ar: Callable[[float], float] = lambda d: 1.0048 + 0.0057 * d - 2.628 * d ** 2 + 3.682 * d ** 3 - \
                                                              1.677 * d ** 4
            ar = andsager_ar(diameters / 10)

        x_ku = 2 * np.pi * (diameters / 2.) / tmatrix_aux.wl_Ku
        x_ka = 2 * np.pi * (diameters / 2.) / tmatrix_aux.wl_Ka
        x_w = 2 * np.pi * (diameters / 2.) / tmatrix_aux.wl_W

        # Tmatrix calculations
        # forward scatterer
        ku_scatter_fw = _scatterer(diameters=diameters, ar=ar, wl='Ku', j=j)
        ka_scatter_fw = _scatterer(diameters=diameters, ar=ar, wl='Ka', j=j)
        w_scatter_fw = _scatterer(diameters=diameters, ar=ar, wl='W', j=j)

        # extinction cross-section
        ku_extinction = [ext_xsect(i) for i in ku_scatter_fw]
        ka_extinction = [ext_xsect(i) for i in ka_scatter_fw]
        w_extinction = [ext_xsect(i) for i in w_scatter_fw]

        # backward scatterer
        ku_scatter_bw = _scatterer(diameters=diameters, ar=ar, wl='Ku', j=j, forward=False)
        ka_scatter_bw = _scatterer(diameters=diameters, ar=ar, wl='Ka', j=j, forward=False)
        w_scatter_bw = _scatterer(diameters=diameters, ar=ar, wl='W', j=j, forward=False)

        # Backscattering cross-section
        ku_bckscatt = [radar.radar_xsect(i) for i in ku_scatter_bw]
        ka_bckscatt = [radar.radar_xsect(i) for i in ka_scatter_bw]
        w_bckscatt = [radar.radar_xsect(i) for i in w_scatter_bw]

        # Mie calculations
        mie_ku = [Mie(x=x_ku[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ku]).qb() * np.pi * (i / 2.) ** 2 for w, i in
                  enumerate(diameters)]
        mie_ka = [Mie(x=x_ka[w], m=refractive.m_w_0C[tmatrix_aux.wl_Ka]).qb() * np.pi * (i / 2.) ** 2 for w, i in
                  enumerate(diameters)]
        mie_w = [Mie(x=x_w[w], m=refractive.m_w_0C[tmatrix_aux.wl_W]).qb() * np.pi * (i / 2.) ** 2 for w, i in
                 enumerate(diameters)]

        df_scatter = pd.DataFrame(
            {'T_mat_Ku': ku_bckscatt, 'T_mat_Ka': ka_bckscatt, 'T_mat_W': w_bckscatt, 'Mie_Ku': mie_ku, 'Mie_Ka': mie_ka,
             'Mie_W': mie_w, 'Ku_extc': ku_extinction, 'Ka_extc': ka_extinction, "W_extc": w_extinction}, index=diameters)
        path_db = f'{path_data}/cloud_probes/db'
        make_dir(path_db)
        str_db = f"sqlite:///{path_db}/scattering_{_lower}_{_upper}.sqlite"
        df_scatter.to_sql(f'{instrument}', con=str_db, if_exists='replace')
        return df_scatter


def radar_calc(nd, _lower, _upper, mie=False):
    ds = np.fromiter(nd.attrs['dsizes'].keys(), dtype=float) / 1e3
    try:
        path_db = f'{path_data}/cloud_probes/db'
        str_db = f"sqlite:///{path_db}/scattering_{_lower}_{_upper}.sqlite"
        backscatter = pd.read_sql(f"{nd.attrs['instrument']}", con=str_db)
    except (OperationalError, ValueError):
        backscatter = bck_extc_crss(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper)

    if len(ds) != backscatter.shape[0]:
        backscatter = bck_extc_crss(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper)

    dsizes = np.fromiter(nd.attrs['dsizes'].values(), dtype=float) / 1e3
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    wvl = ['z_Ku', 'z_Ka', 'z_W', 'A_Ku', 'A_Ka', 'A_W']
    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_Ku'].values,
                                                                     axis='columns').mul(dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_Ka'].values,
                                                                     axis='columns').mul(dsizes)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_W'].values,
                                                                   axis='columns').mul(dsizes)
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_Ku'].values,
                                                                     axis='columns').mul(dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_Ka'].values,
                                                                     axis='columns').mul(dsizes)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_W'].values,
                                                                   axis='columns').mul(dsizes)

    att_ku = (0.01 / np.log10(10)) * nd.mul(1e6).mul(backscatter['Ku_extc'].values,
                                                     axis='columns').mul(dsizes)
    att_ka = (0.01 / np.log10(10)) * nd.mul(1e6).mul(backscatter['Ka_extc'].values,
                                                     axis='columns').mul(dsizes)
    att_w = (0.01 / np.log10(10)) * nd.mul(1e6).mul(backscatter['W_extc'].values,
                                                    axis='columns').mul(dsizes)

    df_radar = pd.concat([z_ku, z_ka, z_w, att_ku, att_ka, att_w], axis=1, keys=wvl, levels=[wvl])
    return df_radar
