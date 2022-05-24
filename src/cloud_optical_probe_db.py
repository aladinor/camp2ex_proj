#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import functools
import operator
import pandas as pd
from re import split, findall
import matplotlib
from pytmatrix import tmatrix_aux, refractive, tmatrix, radar
from pymiecoated import Mie
from scipy.constants import c
matplotlib.use('agg')
import numpy as np

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def vel(d):
    return -0.1021 + 4.932 * d - 0.955 * d ** 2 + 0.07934 * d ** 3 - 0.0023626 * d ** 4


def pds_parameters(nd, d, dd):
    try:
        lwc = (np.pi / 6) * 1e-3 * np.sum(nd * d ** 3 * dd)  # g / m3
        dm = np.sum(nd * d ** 4 * dd) / np.sum(nd * d ** 3 * dd)  # mm
        nw = 1e3 * (4 ** 4 / np.pi) * (lwc / dm ** 4)
        z = np.sum(nd * d ** 6 * dd)
        r = np.pi * 6e-4 * np.sum(nd * d ** 3 * vel(d) * dd)
        ref = ref_calc(nd=nd, d=d, dd=dd)
        return pd.Series([lwc, dm, nw, z, r]).append(ref)
    except ZeroDivisionError:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def bcksct(ds, ar=1, j=0) -> dict:
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
    tmat_ku = [radar.radar_xsect(tmatrix.Scatterer(radius=i/2., wavelength=tmatrix_aux.wl_Ku,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ku], axis_ratio=1.0/ar, thet0=j,
                                                   thet=180 - j,
                                                   phi0=0., phi=180., radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM)) for
               i in ds]
    tmat_ka = [radar.radar_xsect(tmatrix.Scatterer(radius=i / 2., wavelength=tmatrix_aux.wl_Ka,
                                                   m=refractive.m_w_0C[tmatrix_aux.wl_Ka], axis_ratio=1.0/ar, thet0=j,
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

    return {'T_mat_Ku': tmat_ku, 'T_mat_Ka': tmat_ka, 'T_mat_W': tmat_w, 'Mie_Ku': mie_ku, 'Mie_Ka': mie_ka,
            'Mie_W': mie_w}


def ref_calc(nd, d, dd, mie=False):
    backscatter = scatter
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_Ku'] * nd * dd)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_Ka'] * nd * dd)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['Mie_W'] * nd * dd)
        return pd.Series({'Ku': 10 * np.log10(z_ku), 'Ka': 10 * np.log10(z_ka), 'W': 10 * np.log10(z_w)},
                         name=nd.attrs['instrument'])
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['T_mat_Ku'] * nd * dd)
        zku_r = np.sum(nd * (d ** 6) * dd)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['T_mat_Ka'] * nd * dd)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * np.sum(backscatter['T_mat_W'] * nd * dd)
        return pd.Series({'Ku': 10 * np.log10(z_ku), 'Ka': 10 * np.log10(z_ka), 'W': 10 * np.log10(z_w)})


def find_nearest2(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices


def get_merge_cols(col1, cols2, val=1000):
    sizes1 = np.array([float(''.join(findall(r"\d*\.\d+|\d+", i[i.find('(') + 1: i.find(')')])[:1]))
                      for i in col1])
    sizes2 = np.array([float(''.join(findall(r"\d*\.\d+|\d+", i[i.find('(') + 1: i.find(')')])[:1]))
                       for i in cols2])
    return col1[:find_nearest2(sizes1, val)], cols2[find_nearest2(sizes2, val):]


def get_sizes(cols):
    sizes = np.array([float(''.join(findall(r"\d*\.\d+|\d+", i[i.find('(') + 1: i.find(')')])[:1]))
                      for i in cols])
    dsizes = sizes[1:] - sizes[:-1]
    dsizes = np.append(dsizes, dsizes[-1])
    bin_cent = (sizes[1:] - sizes[:-1]) / 2 + sizes[:-1]
    bin_cent = np.append(bin_cent, sizes[-1] + dsizes[-1])
    dt_sizes = {i: j for i, j in zip(bin_cent, dsizes)}
    return dt_sizes


def get_cols(table_name, str_db):
    _cols = pd.read_sql_query('''SELECT c.name 
                                    FROM pragma_table_info('{}') c 
                                    WHERE name like 'nsd%'
                                 '''.format(table_name),
                              con=str_db).values
    str_cols = [f"{table_name.split('_')[-1].upper()[:3]}.{i}" for i in functools.reduce(operator.iconcat, _cols, [])]
    return str(str_cols).replace('[', '').replace(']', '')


def get_nsd_data(table_name, str_db, cols):
    print(table_name, '----------------------------', str_db)
    if table_name.split('_')[-1].upper()[:3] == 'LEA':
        str_data = '''
                    SELECT {}, {}', pl."Temp"
                    FROM "{}" {}
                    INNER JOIN "Page0_Learjet" pl ON pl.time = LEA.time
                    WHERE pl."Temp" >= 2
                   '''.format(f"{table_name.split('_')[-1].upper()[:3]}.time",
                              cols.replace("'", "").replace(table_name.split('_')[-1].upper()[:3] + ".",
                                                            table_name.split('_')[-1].upper()[:3] +
                                                            "." + "'").replace(",", "',"),
                              table_name, table_name.split('_')[-1].upper()[:3])
        return pd.read_sql_query(str_data, con=str_db)
    else:
        str_data = '''
                    SELECT {}, {}', pbm.' Static_Air_Temp_YANG_MetNav' as "Temp" 
                    FROM "{}" {}
                    INNER JOIN p3b_merge pbm ON pbm.time = P3B.time
                    Where Temp >= 2
                   '''.format(f"{table_name.split('_')[-1].upper()[:3]}.time",
                              cols.replace("'", "").replace(table_name.split('_')[-1].upper()[:3] + ".",
                                                            table_name.split('_')[-1].upper()[:3] + "."
                                                            + "'").replace(",", "',"),
                              table_name, table_name.split('_')[-1].upper()[:3])
        return pd.read_sql_query(str_data, con=str_db)


def main():
    path_db = f'{path_data}/db'
    str_db = f"sqlite:///{path_db}/camp2ex.sqlite"
    instruments = ['2DS10', 'HVPS']
    aircraft = ['Learjet', 'P3B']
    ls_cols = {air: {instrument: get_cols(f'{instrument}_{air}', str_db) for instrument in instruments} for air in
               aircraft}

    #  LEARJET - 2DS10
    df_lear_2ds = get_nsd_data(table_name=f"{instruments[0]}_{aircraft[0]}", str_db=str_db,
                               cols=f"{ls_cols[aircraft[0]][instruments[0]]}")
    df_lear_2ds['time'] = pd.to_datetime(df_lear_2ds['time'])
    df_lear_2ds.set_index(df_lear_2ds['time'], inplace=True, drop=True)

    #  LEARJET - HVPS
    df_lear_hvps = get_nsd_data(table_name=f"{instruments[1]}_{aircraft[0]}", str_db=str_db,
                                cols=f"{ls_cols[aircraft[0]][instruments[1]]}")
    df_lear_hvps['time'] = pd.to_datetime(df_lear_hvps['time'])
    df_lear_hvps.set_index(df_lear_hvps['time'], inplace=True, drop=True)
    col_2ds, col_hvps = get_merge_cols(df_lear_2ds.filter(like='nsd').columns, df_lear_hvps.filter(like='nsd').columns)
    df_merged = pd.merge(df_lear_2ds[col_2ds], df_lear_hvps[col_hvps], right_index=True, left_index=True)
    sizes_merged = get_sizes(list(df_merged.filter(like='nsd').columns))
    d_merged = np.fromiter(sizes_merged.keys(), dtype=float) / 1e3
    dd_merged = np.fromiter(sizes_merged.values(), dtype=float)
    global scatter
    scatter = bcksct(ds=d_merged)

    df_merged[['lwc', 'dm', 'nw', 'z', 'r', 'Ku_ref', 'Ka_ref', 'W_ref']] = \
        df_merged.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values * 1e3, d=d_merged, dd=dd_merged),
                          axis=1)
    str_db_save = f"sqlite:///{path_db}/merged.sqlite"
    df_merged.to_sql(f'2ds_hvps_lear', con=str_db_save, if_exists='replace')
    del df_merged, df_lear_hvps, df_lear_2ds

    #  P3B - 2DS10
    df_p3b_2ds = get_nsd_data(table_name=f"{instruments[0]}_{aircraft[1]}", str_db=str_db,
                               cols=f"{ls_cols[aircraft[1]][instruments[0]]}")
    df_p3b_2ds['time'] = pd.to_datetime(df_p3b_2ds['time'])
    df_p3b_2ds.set_index(df_p3b_2ds['time'], inplace=True, drop=True)

    #  LEARJET - HVPS
    df_p3b_hvps = get_nsd_data(table_name=f"{instruments[1]}_{aircraft[1]}", str_db=str_db,
                                cols=f"{ls_cols[aircraft[1]][instruments[1]]}")
    df_p3b_hvps['time'] = pd.to_datetime(df_p3b_hvps['time'])
    df_p3b_hvps.set_index(df_p3b_hvps['time'], inplace=True, drop=True)
    col_2ds, col_hvps = get_merge_cols(df_p3b_2ds.filter(like='nsd').columns, df_p3b_hvps.filter(like='nsd').columns)
    df_merged = pd.merge(df_p3b_2ds[col_2ds], df_p3b_hvps[col_hvps], right_index=True, left_index=True)
    sizes_merged = get_sizes(list(df_merged.filter(like='nsd').columns))
    d_merged = np.fromiter(sizes_merged.keys(), dtype=float) / 1e3
    dd_merged = np.fromiter(sizes_merged.values(), dtype=float)
    df_merged[['lwc', 'dm', 'nw', 'z', 'r', 'Ku_ref', 'Ka_ref', 'W_ref']] = \
        df_merged.apply(lambda x: pds_parameters(nd=x.filter(like='nsd').values * 1e3, d=d_merged, dd=dd_merged),
                          axis=1)
    df_merged.to_sql(f'2ds_hvps_p3b', con=str_db_save, if_exists='replace')


if __name__ == "__main__":
    main()
