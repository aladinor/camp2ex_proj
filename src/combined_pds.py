#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
from typing import Callable
from scipy.special import gamma
import numpy as np
import pandas as pd
from sqlalchemy.exc import OperationalError
import xarray as xr
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


def change_cols(df):
    bin_cent = df.attrs['bin_cent']
    cols = df.columns
    new_cols = {cols[i]: bin_cent[i] for i in range(len(cols))}
    df = df.rename(columns=new_cols)
    return df


def filter_by_cols(ls_df):
    # if cols:
    cols = [[j for j in i.columns if j.startswith('nsd')] for i in ls_df]
    ls_df = [i[cols[j]] for j, i in enumerate(ls_df)]
    ls_df = [change_cols(i) for i in ls_df]
    return ls_df


def get_intervals(ls_df):
    cols = [[pd.Interval(j.attrs['sizes'][i], j.attrs['sizes'][i + 1]) for i in
             range(len(j.attrs['sizes']) - 1)] for j in ls_df]
    cols = [j + [pd.Interval(ls_df[i].attrs['sizes'][-1], list(ls_df[i].attrs['dsizes'].keys())[-1])]
            for i, j in enumerate(cols)]
    return cols


def apply_wgt(df, ovr_upp, ovr_lower):
    if (df['2DS10'] != 0) & (df['HVPS'] != 0):
        df['2ds10_wgt'] = df['2DS10'] * df['2DS10'].index.values / (ovr_upp - ovr_lower)
        df['hvps_wgt'] = df['HVPS'] * (df['HVPS'].index.values - ovr_lower) / (ovr_upp - ovr_lower)
        df['nd_res'] = df[['2ds10_wgt', 'hvps_wgt']].dropna(how='all').sum(1)
        return df['nd_res']
    elif (df['2DS10'] == 0) & (df['HVPS'] != 0):
        return df['HVPS']
    elif (df['2DS10'] != 0) & (df['HVPS'] == 0):
        return df['HVPS']
    else:
        return np.nan


def linear_wgt(df1, df2, ovr_upp=1200, ovr_lower=800, method='linear', lower_limit=50, upper_limit=4000):
    """
    Functions that perform linear weight interpolation between to cloud probes
    :param upper_limit: Upper limit until data is used in um
    :param lower_limit: Lower limit until data is used in um
    :param method: method to apply. linear applies Leroy et al. 2014. swal Applies Snesbitt method
    :param df1: pandas series with the small diameter sizes (e.g. 2DS10)
    :param df2: pandas series with the small diameter sizes (e.g. HVPS)
    :param ovr_upp: upper limit for overlapping
    :param ovr_lower: lower limit for overlapping
    :return: pd series with a composite PSD
    """
    comb_upper = ovr_upp + 200
    comb_lower = ovr_lower - 50
    df1.columns = df1.attrs['2DS10']['intervals']
    df2.columns = df2.attrs['HVPS']['intervals']
    cond1 = (df1.columns.mid >= comb_lower) & (df1.columns.mid <= comb_upper)
    cond2 = (df2.columns.mid >= comb_lower) & (df2.columns.mid <= comb_upper)
    cond1_merg = (df1.columns.mid >= ovr_lower) & (df1.columns.mid <= ovr_upp)
    _nd_uppr = df1.iloc[:, cond1]
    _nd_lower = df2.iloc[:, cond2]
    instr = ['2DS10', 'HVPS']
    nd_overlap = pd.concat([_nd_uppr.reindex(columns=np.arange(comb_lower, comb_upper, 5)),
                            _nd_lower.reindex(columns=np.arange(comb_lower, comb_upper, 5))], axis=1, keys=instr,
                           levels=[instr])
    if method == 'linear':
        nd_overlap['2ds10_wgt'] = nd_overlap['2DS10'] * (ovr_upp - nd_overlap['2DS10'].columns.values) / \
                                  (ovr_upp - ovr_lower)
        nd_overlap['hvps_wgt'] = nd_overlap['HVPS'] * (nd_overlap['HVPS'].index.values - ovr_lower) / \
                                 (ovr_upp - ovr_lower)
        nd_overlap['nd_res'] = nd_overlap[['2ds10_wgt', 'hvps_wgt']].dropna(how='all').sum(1)
        nd_overlap = nd_overlap.reindex(df1[cond1].index.mid)
        nd_overlap.index = df1[cond1].index
        res = pd.concat([df1[df1.index.mid <= ovr_lower], nd_overlap['nd_res'], df2[df2.index.mid >= ovr_upp]])
        d_d = {i.mid: i.length for i in res.index}
        res.index = res.index.mid
        res.attrs['dsizes'] = d_d
        return res

    elif 'snal':
        _sdf = nd_overlap.stack()[(((nd_overlap.stack()['2DS10'] == 0) & (nd_overlap.stack()['2DS10'].notnull())) &
                                   ((nd_overlap.stack()['HVPS'] == 0) & (nd_overlap.stack()['HVPS'].notnull())))][
            '2DS10'].unstack()
        _sdd = nd_overlap.stack()[(((nd_overlap.stack()['2DS10'] != 0) & (nd_overlap.stack()['2DS10'].notnull())) &
                                   ((nd_overlap.stack()['HVPS'] != 0) & (nd_overlap.stack()['HVPS'].notnull())))]
        _sdd['2ds10_wgt'] = _sdd['2DS10'] * (ovr_upp - _sdd.index.get_level_values(1)) / (ovr_upp - ovr_lower)
        _sdd['hvps_wgt'] = _sdd['HVPS'] * (_sdd.index.get_level_values(1) - ovr_lower) / (ovr_upp - ovr_lower)
        _sdd['nd_res'] = _sdd[['2ds10_wgt', 'hvps_wgt']].sum(1)

        _sdd = _sdd['nd_res'].unstack()
        _sdc = nd_overlap.stack()[(((nd_overlap.stack()['2DS10'] == 0) | (nd_overlap.stack()['2DS10'].isnull())) &
                                   (nd_overlap.stack()["HVPS"] != 0))]['HVPS'].unstack()
        _sds = nd_overlap.stack()[(((nd_overlap.stack()['HVPS'] == 0) | (nd_overlap.stack()['HVPS'].isnull())) &
                                   (nd_overlap.stack()["2DS10"] != 0))]['2DS10'].unstack()
        res = pd.concat([_sdd, _sdc, _sds, _sdf], axis=1).sort_index().dropna(how='all',
                                                                              axis='columns').groupby(level=0,
                                                                                                      axis=1).sum()
        res = res[df1.iloc[:, cond1_merg].columns.mid]
        res.columns = df1.iloc[:, cond1_merg].columns
        res = pd.concat([df1.iloc[:, df1.columns.mid < ovr_lower], res, df2.iloc[:, df2.columns.mid > ovr_upp]],
                        axis=1)
        res = res.iloc[:, (res.columns.mid >= lower_limit) & (res.columns.mid <= upper_limit)]
        d_d = {i.mid: i.length for i in res.columns}
        res.columns = res.columns.mid
        res.attrs['dsizes'] = d_d
        res.attrs['instrument'] = 'Composite_PSD'
        return res


def pds_parameters(nd, vel="lhermitte"):
    """
    Compute the psd parameters
    :param nd: partice size distribution in # L-1 um-1
    :param vel: equation used for terminal velocity stimation. By default Lhermitte.
    :return: list with lwc, dm, nw, z, and r
    """
    lerm_vel: Callable[[float], float] = lambda diam: 9.25 * (1 - np.exp(-0.068 * diam ** 2 - 0.488 * diam))  # d in mm
    ulbr_vel: Callable[[float], float] = lambda diam: 3.78 * diam ** 0.67  # with d in mm

    d = pd.DataFrame(data=np.tile(np.fromiter(nd.columns, dtype=float) / 1e3, (nd.shape[0], 1)), index=nd.index,
                     columns=nd.columns)
    d_d = pd.DataFrame(data=np.tile(np.fromiter(nd.attrs['dsizes'].values(), dtype=float) / 1e3,
                                    (nd.shape[0], 1)), index=nd.index, columns=nd.columns)
    if vel == "lhermitte":
        vel = lerm_vel(d)
    else:
        vel = ulbr_vel(d)

    nt = nd.mul(1e6).mul(d_d).sum(1)
    lwc = nd.mul(1e6).mul(d ** 3).mul(d_d) * (np.pi / (6 * 1000))  # g / m3
    dm = nd.mul(1e6).mul(d ** 4).mul(d_d).sum(1) / nd.mul(1e6).mul(d ** 3).mul(d_d).sum(1)  # mm
    nw = 1e3 * (4 ** 4 / np.pi) * (lwc.sum(1) / dm ** 4)
    z = nd.mul(1e6).mul(d ** 6).mul(d_d)
    r = 6 * np.pi * 1e-4 * (nd.mul(1e6).mul(d ** 3).mul(d_d).mul(vel)).sum(1)
    sigmasqr = d.sub(dm, axis='rows').pow(2).mul(nd * 1e6 * d ** 3 * d_d).sum(1) / (nd * 1e6 * d ** 3 * d_d).sum(1)
    sigma = np.sqrt(sigmasqr)
    br = np.arange(0.5, 5.5, 0.001)
    mu = dm ** 2 / sigmasqr - 4
    _ = ['nt', 'lwc', 'dm', 'nw', 'z', 'r', 'sigmasqr', 'sigma', 'mu']
    df = pd.concat([nt, lwc, dm, nw, z, r, sigmasqr, sigma, mu], axis=1, keys=_, levels=[_])
    res = np.zeros_like(br)
    for i in range(br.shape[0]):
        res[i] = np.corrcoef(dm, sigma / dm ** br[i])[0, 1] ** 2
    bm = br[np.argmin(res)]
    df['sigma_prime'] = sigma.values / (dm.values ** bm)
    df['new_sigma'] = df['sigma_prime'].mean() * df['dm'] ** bm
    df['new_mu'] = (dm.values ** (2 - 2 * bm) / (df['sigma_prime'].mean() ** 2)) - 4
    df['mu_williams'] = 11.1 * df['dm'] ** (-0.72) - 4
    df['mu_camp2ex'] = ((df['dm'] ** (2 - 2 * bm)) / (df['sigma_prime'].mean() ** 2)) - 4
    return df


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


def bck_extc_crss(diameters, instrument, _lower, _upper, ar=None, j=0) -> pd.DataFrame:
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
    if ar is None:
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
        ar = np.ones_like(ds)
        backscatter = bck_extc_crss(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper, ar=ar)

    if len(ds) != backscatter.shape[0]:
        ar = np.ones_like(ds)
        backscatter = bck_extc_crss(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper, ar=ar)

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


def compute_hr(t, td):
    """
    Computes relative humidity using magnus approximation using
    https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    :param t: temperature
    :param td: dew point temperature
    :return: relative humidity
    """
    b = 17.625
    _c = 243.04
    return 100 * np.exp((_c * b * (td - t) / ((_c + t) * (_c + td))))


def get_add_data(aircraft: 'str', indexx) -> pd.DataFrame:
    path_par = f'{path_data}/cloud_probes/pkl'
    if aircraft == 'Lear':
        str_db = f'{path_par}/Page0_Learjet.pkl'
        df_add = pd.read_pickle(str_db)
        df_add = df_add.filter(['Temp', 'Dew', 'Palt', 'NevLWC', 'VaV', "Lat", "Long"])
        df_add['RH'] = compute_hr(t=df_add['Temp'], td=df_add['Dew'])
        df_add['RH'] = df_add['RH'].where(df_add['RH'] <= 100, 100)
        cols = ['temp', 'dew_point', 'altitude', 'lwc', 'vertical_vel', 'lat', 'lon', 'RH']
        new_cols = {j: cols[i] for i, j in enumerate(list(df_add.columns))}
        df_add.rename(columns=new_cols, inplace=True)
        df_add = df_add[(df_add.index >= indexx.min().strftime('%Y-%m-%d %X')) &
                        (df_add.index <= indexx.max().strftime('%Y-%m-%d %X'))]
        return df_add
    else:
        str_db = f'{path_par}/p3b_merge.pkl'
        df_add = pd.read_pickle(str_db)
        df_add = df_add.filter([' Total_Air_Temp_YANG_MetNav', ' Dew_Point_YANG_MetNav',
                                ' GPS_Altitude_YANG_MetNav', ' LWC_gm3_LAWSON',
                                ' Vertical_Speed_YANG_MetNav', ' Relative_Humidity_YANG_MetNav',
                                ' Roll_Angle_YANG_MetNav', ' Latitude_YANG_MetNav', ' Longitude_YANG_MetNav'])
        cols = ['temp', 'dew_point', 'altitude', 'lwc', 'vertical_vel', 'RH', 'roll', 'lat', 'lon']
        new_cols = {j: cols[i] for i, j in enumerate(list(df_add.columns))}
        df_add.rename(columns=new_cols, inplace=True)
        df_add.roll[df_add.roll > 180] = df_add.roll[df_add.roll > 180] - 360
        df_add = df_add[(df_add.index >= indexx.min().strftime('%Y-%m-%d %X')) &
                        (df_add.index <= indexx.max().strftime('%Y-%m-%d %X'))]
        return df_add


def area_filter(ds):
    diameter = ds.attrs['bin_cent'] / 1e3  # mm
    andsager_ar: Callable[[float], float] = lambda d: 1.0048 + 0.0057 * d - 2.628 * d ** 2 + 3.682 * d ** 3 - \
                                                      1.677 * d ** 4
    ar = andsager_ar(diameter / 10)
    area_func: Callable[[float], float] = lambda x: np.pi * (x / 2) ** 2
    area = area_func(diameter) / 1e5
    _lower = area * ar
    _upper = area * ar + area * ar * 2
    avg_area = ds.filter(like='a_bin').values / ds.filter(like='cnt').values
    df_area = pd.DataFrame(avg_area, index=ds.index, columns=ds.filter(like='nsd').columns, dtype='float64')
    ds_area = df_area[(df_area >= _lower) & (df_area <= _upper)].notnull()
    cols = ds.filter(like='nsd').columns
    ds.loc[:, cols] = ds.filter(like='nsd')[ds_area]
    return ds


def fill_2ds(ls_df):
    df_2ds = [i for i in ls_df if i.attrs['instrument'] == '2DS10'][0]
    df_h2ds = [i for i in ls_df if i.attrs['instrument'] == 'Hawk2DS10'][0]
    df_2ds = df_2ds.fillna(df_h2ds)
    ls_df = [df_2ds if i.attrs['instrument'] == '2DS10' else i for i in ls_df]
    return ls_df


def filter_by_bins(df, nbins=10, dt=None):
    """
    Function that filter a pandas dataframe row with less than nbins of consecutive data
    :param dt: dictionary with lowwer and upper limits
    :param df: pandas dataframe with psd data.
    :param nbins: number of consecutive bins
    :return: filtered dataframe
    """
    if not dt:
        dt = {'2DS10': {'low': 50., 'up': 1000.},
              'Hawk2DS10': {'low': 50., 'up': 1000.},
              'HVPS': {'low': 300, 'up': 4000.}}
        cols = df.columns
        new_cols = [i for i in cols if (i >= dt[df.attrs['instrument']]['low'])]
        new_cols = [i for i in new_cols if (i <= dt[df.attrs['instrument']]['up'])]
    else:
        new_cols = df.columns
    df_copy = df[new_cols]
    df_ones = df_copy.replace([-9.99, 0], np.nan).notnull().astype(int)
    df_cum = df_ones.cumsum(1).replace(0, np.nan)
    _reset = -df_cum[df.replace([-9.99, 0], np.nan).isnull()].fillna(method='pad', axis=1). \
        diff(axis=1).replace(0, np.nan).fillna(df_cum)
    res = df_ones.where(df.replace([-9.99, 0], np.nan).notnull(), _reset).cumsum(1) # where counter rest
    _max = res[res > 0].max(axis=1)
    df['nbins'] = _max
    df = df[df['nbins'] >= nbins]
    df = df.drop(['nbins'], axis=1)
    del df_ones, df_cum, _reset, res, df_copy
    return df


def filt_by_roll(df, roll=5):
    """
    Function that filters data out by plane roll.
    :param df: pandas dataframe to be filtered
    :param roll: maximum roll angle. e.g. 5 deg
    filtered dataframe
    """
    return df[(df['roll'] > -roll) & (df['roll'] < roll)]


def norm_gamma(d, nw, mu, dm):
    f_mu = (6 * (4 + mu) ** (mu + 4)) / (4 ** 4 * gamma(mu.astype('float') + 4))
    slope = (4 + mu) / dm
    return nw * f_mu * (d / dm) ** mu * np.exp((-slope * d).astype('float'))


def ref_gamma(ds_gm, prefix, d_d, _lower=600, _upper=1000, mie=False, instrument='Composite_PSD', onlyref=False):
    try:
        path_db = f'{path_data}/cloud_probes/db'
        str_db = f"sqlite:///{path_db}/scattering_{_lower}_{_upper}.sqlite"
        backscatter = pd.read_sql(f"{instrument}", con=str_db)
    except (OperationalError, ValueError):
        ar = np.ones_like(ds_gm.diameter.values)
        backscatter = bck_extc_crss(ds_gm, instrument=instrument, _lower=_lower, _upper=_upper, ar=ar)

    if len(ds_gm.diameter) != backscatter.shape[0]:
        ar = np.ones_like(ds_gm)
        backscatter = bck_extc_crss(ds_gm, instrument=instrument, _lower=_lower, _upper=_upper, ar=ar)

    backscatter = backscatter.reset_index().drop(columns=['level_0', 'index']).\
        assign(diameter=ds_gm.diameter).set_index('diameter').to_xarray()
    dsizes = d_d / 1000
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['Mie_Ku'] * dsizes
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['Mie_Ka'] * dsizes
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['Mie_W'] * dsizes
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['T_mat_Ku'] * dsizes
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['T_mat_Ka'] * dsizes
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * ds_gm * backscatter['T_mat_W'] * dsizes
    if onlyref:
        return z_ku
    else:
        att_ku = (0.01 / np.log10(10)) * ds_gm * backscatter['Ku_extc'] * dsizes
        att_ka = (0.01 / np.log10(10)) * ds_gm * backscatter['Ka_extc'] * dsizes
        att_w = (0.01 / np.log10(10)) * ds_gm * backscatter['W_extc'] * dsizes
        z_ku, z_ka, z_w = [10 * np.log10(i.sum('diameter').astype(float)) for i in [z_ku, z_ka, z_w]]
        new_ds = z_ku.to_dataset(name=f'z_ku_{prefix}')
        new_ds[f'z_ka_{prefix}'] = z_ka
        new_ds[f'z_w_{prefix}'] = z_w
        new_ds[f'att_ku_{prefix}'] = att_ku
        new_ds[f'att_ka_{prefix}'] = att_ka
        new_ds[f'att_w_{prefix}'] = att_w
        new_ds[f'dfr_{prefix}'] = new_ds[f'att_ku_{prefix}'] - new_ds[f'att_ka_{prefix}']
        return new_ds


def radar_from_gamma(d, nw, dm, mu, d_d, prefix):
    ng = norm_gamma(d / 1000,
                    nw=nw,
                    dm=dm,
                    mu=mu)
    return ref_gamma(ds_gm=ng, d_d=d_d, prefix=prefix)


def mn(ds, n=0):
    return ((ds.psd * 1e6) ** n * (ds.diameter / 1000) * ds.d_d).sum('diameter')


def mu_retrieval(ds):
    eta = (mn(ds, n=1) ** 2 / (mn(ds, 0) * mn(ds, 2)))
    return 1 / (1 - eta) - 2


def mu_root(ds, mus):
    gm = norm_gamma(d=ds.diameter/1000, nw=ds.nw, dm=ds.dm, mu=mus)
    y = ds.psd * 1e6
    x = gm.astype(float)
    norm = (y - y.mean('diameter')) / y.std('diameter')
    return (np.log10(y - x) / norm).sum('diameter')


def root(ref_diff, mus):
    try:
        idmin = np.nanargmin(np.abs(ref_diff - 0.0))
        return mus[idmin]
    except ValueError:
        return np.nan


def wrapper(ref_diff, mus):
    return xr.apply_ufunc(
        root,
        ref_diff,
        mus,
        input_core_dims=[['mu'], ['mu']],
        vectorize=True,
        dask='Parallelized'
    )


def main():
    _bef = True
    aircraft = ['Lear', 'P3B']
    for air in aircraft:
        intervals = [600, 1000]
        for nbin in np.arange(1, 10, 1):
            _lower = intervals[0]
            _upper = intervals[-1]
            ls_df = get_data(air, temp=2, sensors=['2DS10', 'HVPS', 'Hawk2DS10'])
            ls_df = fill_2ds(ls_df)
            hvps = area_filter([i for i in ls_df if i.attrs['instrument'] == 'HVPS'][0])
            hvps = hvps[hvps.twc > 0.05]
            ls_df = [hvps if i.attrs['instrument'] == 'HVPS' else i for i in ls_df]
            ls_df = [i[i.twc > 0.05] if i.attrs['instrument'] == '2DS10' else i for i in ls_df]
            ls_df = filter_by_cols(ls_df)
            instr = [i.attrs['instrument'] for i in ls_df]
            if _bef is True:
                ls_df = [filter_by_bins(i, nbins=nbin) for i in ls_df if i.attrs['instrument'] in ['2DS10', 'HVPS']]
            attrs = [i.attrs for i in ls_df]
            dt_attrs = {instr[i]: j for i, j in enumerate(attrs)}
            for idx, att in enumerate(attrs):
                ls_df[idx].attrs = attrs[0]
            df_concat = pd.concat(ls_df, axis=1, keys=instr, levels=[instr])
            df_concat.attrs = dt_attrs

            if location in ['atmos', 'alfonso']:
                if air == "Lear":
                    indexx = pd.date_range(start='2019-09-07 2:31:45', periods=150, tz='UTC', freq='S')  # for Lear
                else:
                    indexx = pd.date_range(start='2019-08-27 00:15', periods=100000, tz='UTC', freq='S')  # for P3B
            else:
                indexx = df_concat.index

            df_concat = df_concat[(df_concat.index >= f"{indexx.min()}") & (df_concat.index <= f"{indexx.max()}")]
            df_merged = linear_wgt(df_concat['2DS10'], df_concat['HVPS'], ovr_upp=intervals[-1], ovr_lower=intervals[0],
                                   method='snal')
            df_merged[df_merged < 0] = np.nan
            if air == "Lear":
                df_merged = df_merged[~(df_merged == '2019-09-09 00:54:08')]
            if _bef is not True:
                df_merged = filter_by_bins(df_merged, nbins=nbin, dt='Combined_PSD')
            df_reflectivity = radar_calc(df_merged, _upper=_upper, _lower=_lower)
            params = pds_parameters(df_merged)
            df_add = get_add_data(air, indexx=indexx)
            d_d = np.fromiter(df_merged.attrs['dsizes'].values(), dtype=float)
            df_merged = df_merged.join(df_add)
            ncols = 8
            if air == "P3B":
                df_merged = filt_by_roll(df_merged, roll=7)
                df_reflectivity = df_reflectivity.loc[df_merged.index]
                params = params.loc[df_merged.index]
                ncols = 9
            xr_merg = xr.Dataset(
                data_vars=dict(
                    psd=(["time", "diameter"], df_merged[df_merged.columns[:-ncols]].to_numpy()),
                    refl_ku=(["time", "diameter"], df_reflectivity['z_Ku'].to_numpy()),
                    refl_ka=(["time", "diameter"], df_reflectivity['z_Ka'].to_numpy()),
                    refl_w=(["time", "diameter"], df_reflectivity['z_W'].to_numpy()),
                    dbz_t_ku=(["time"], 10 * np.log10(df_reflectivity['z_Ku'].sum(1))),
                    dbz_t_ka=(["time"], 10 * np.log10(df_reflectivity['z_Ka'].sum(1))),
                    dbz_t_w=(["time"], 10 * np.log10(df_reflectivity['z_W'].sum(1))),
                    dfr=(["time"], 10 * np.log10(df_reflectivity['z_Ku'].sum(1)) -
                         10 * np.log10(df_reflectivity['z_Ka'].sum(1))),
                    A_ku=(["time", "diameter"], df_reflectivity['A_Ku'].to_numpy()),
                    A_ka=(["time", "diameter"], df_reflectivity['A_Ka'].to_numpy()),
                    A_w=(["time", "diameter"], df_reflectivity['A_W'].to_numpy()),
                    Att_ku=(["time"], 10 * np.log10(df_reflectivity['A_Ku'].sum(1))),
                    Att_ka=(["time"], 10 * np.log10(df_reflectivity['A_Ka'].sum(1))),
                    Att_w=(["time"], 10 * np.log10(df_reflectivity['A_W'].sum(1))),
                    nt=(["time"], params['nt'].to_numpy()[:, 0]),
                    lwc=(["time", "diameter"], params['lwc'].to_numpy()),
                    lwc_cum=(["time"], params['lwc'].sum(1).to_numpy()),
                    mu=(["time"], params['mu'].to_numpy()[:, 0]),
                    new_mu=(["time"], params['new_mu'].to_numpy()),
                    nw=(["time"], params['nw'].to_numpy()[:, 0]),
                    log10_nw=(["time"], np.log10(params['nw'].to_numpy()[:, 0])),
                    dm=(["time"], params['dm'].to_numpy()[:, 0]),
                    z=(["time", "diameter"], params['z'].to_numpy()),
                    r=(["time"], params['r'].to_numpy()[:, 0]),
                    sigmasqr=(["time"], params['sigmasqr'].to_numpy()[:, 0]),
                    sigma=(["time"], params['sigma'].to_numpy()[:, 0]),
                    sigmap=(["time"], params['sigma_prime'].to_numpy()),
                    new_sigma=(["time"], params['new_sigma'].to_numpy()),
                    mu_williams=(["time"], params["mu_williams"].to_numpy()),
                    mu_camp2ex=(["time"], params["mu_camp2ex"].to_numpy()),
                    temp=(["time"], df_merged['temp'].to_numpy()),
                    dew_point=(["time"], df_merged['dew_point'].to_numpy()),
                    altitude=(["time"], df_merged['altitude'].to_numpy()),
                    lwc_plane=(["time"], df_merged['lwc'].to_numpy()),
                    vert_vel=(["time"], df_merged['vertical_vel'].to_numpy()),
                    lat=(["time"], df_merged['lat'].to_numpy()),
                    lon=(["time"], df_merged['lon'].to_numpy()),
                    RH=(["time"], df_merged['RH'].to_numpy()),
                    d_d=(["diameter"], d_d)
                ),
                coords=dict(
                    time=(["time"], np.array([i.to_datetime64() for i in df_merged.index])),
                    diameter=(["diameter"], df_merged.columns[:-ncols])),
                attrs={'combined_pds': 'units: # l um-1',
                       'diameter': 'units # mm',
                       'time': 'UTC',
                       'reflectivity_Ku': 'units mm6 mm-3',
                       'reflectivity_Ka': 'units mm6 mm-3',
                       'reflectivity_W': 'units mm6 mm-3',
                       'LWC': 'units gm-3',
                       'temp': 'units Celcius',
                       'dew_point': 'units Celcius',
                       'altitude': 'units Lear (ft), P3B (m)',
                       'lwc_plane': 'units g m-3, method Lear-NevLWC, P3B-LAWSON',
                       'vert_vel': 'units ms-1',
                       'RH': 'method Lear-Computed, P3B-Measured',
                       'd_d': 'bin lenght in mm'
                       },
            )
            # removing where mu is + inf
            xr_merg = xr_merg.where(xr_merg.mu < 150, drop=True)

            # retrieving radar variables using mu using williams et al 2014 eq. 18
            nw_ds = radar_from_gamma(d=xr_merg.diameter, dm=xr_merg.dm, nw=xr_merg.nw, mu=xr_merg.mu,
                                     d_d=xr_merg.d_d, prefix='mu1')
            xr_merg = xr_merg.merge(nw_ds)

            # retrieving radar variables using mu using williams et al 2014 eq. 25
            nw_ds = radar_from_gamma(d=xr_merg.diameter, dm=xr_merg.dm, nw=xr_merg.nw, mu=xr_merg.new_mu,
                                     d_d=xr_merg.d_d, prefix='mu2')
            xr_merg = xr_merg.merge(nw_ds)

            # retrieving mu parameters using moments method ##M012
            mu = mu_retrieval(xr_merg)
            xr_merg['mu3'] = mu

            # retrieving radar variables using mu using moment M012
            nw_ds = radar_from_gamma(d=xr_merg.diameter, dm=xr_merg.dm, nw=xr_merg.nw, mu=mu,
                                     d_d=xr_merg.d_d, prefix='mu3')
            xr_merg = xr_merg.merge(nw_ds)

            # retrieving mu using brute force
            mu = np.arange(-3.5, 10, 0.05)
            mus = xr.DataArray(data=mu,
                               dims=['mu'],
                               coords=dict(mu=(['mu'], mu)))
            ref_diff = mu_root(xr_merg, mus)
            mu_bf = wrapper(ref_diff.load(), mus)
            xr_merg['mu_bf'] = mu_bf
            nw_ds = radar_from_gamma(d=xr_merg.diameter, dm=xr_merg.dm, nw=xr_merg.nw, mu=mu_bf,
                                     d_d=xr_merg.d_d, prefix='mu_bf')
            xr_merg = xr_merg.merge(nw_ds)
            xr_mean = xr_merg.rolling(time=5).mean()
            if _bef is True:
                store = f"{path_data}/cloud_probes/zarr/combined_psd_{air}_{_lower}_{_upper}_{nbin}_bins.zarr"
                store2 = f"{path_data}/cloud_probes/zarr/combined_psd_{air}_{_lower}_{_upper}_{nbin}_bins_5s.zarr"
                xr_merg.to_zarr(store=store, consolidated=True)
                xr_mean.to_zarr(store=store2, consolidated=True)
            else:
                store = f"{path_data}/cloud_probes/zarr/combined_psd_{air}_{_lower}_{_upper}_{nbin}_bins_merged_5s.zarr"
                store2 = f"{path_data}/cloud_probes/zarr/combined_psd_{air}_{_lower}_{_upper}_{nbin}_bins_merged_5s.zarr"
                xr_merg.to_zarr(store=store, consolidated=True)
                xr_mean.to_zarr(store=store2, consolidated=True)

            del xr_merg, xr_mean
            print(f'done {nbin}')


if __name__ == '__main__':
    main()
