#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
import pandas as pd
from sqlalchemy.exc import OperationalError
import xarray as xr
from pytmatrix import tmatrix_aux, refractive, tmatrix, radar
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


def linear_wgt(df1, df2,  ovr_upp=1200, ovr_lower=800, method='linear'):
    """

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
    cond2_merg = (df2.columns.mid >= ovr_lower) & (df2.columns.mid <= ovr_upp)
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
        d_d = {i.mid: i.length for i in res.columns}
        res.columns = res.columns.mid
        res.attrs['dsizes'] = d_d
        res.attrs['instrument'] = 'Composite_PSD'
        return res


def linear_wgt2(df, intervals: list[int], method='snal'):
    """

    :type intervals: list
    :param method: str. linear applies Leroy et al. 2014. swal Applies Snesbitt method
    :param df: pd.DataFrame. concatenate pandas dataframe with level and keys eg (pd.concat([df_2ds, df_hvps]),
               levels=['2DS10', 'HVPS'], keys=['2DS10', 'HVPS'], axis=1). This dataframe must contain time in rows and
               in columns the diameters
    :param intervals: overlapping intervals
    :return: Combined PSD dataframe
    """
    _intervals = pd.IntervalIndex.from_breaks(intervals)
    df_left = df['FCDP']
    df_left.columns = df_left.attrs['FCDP']['intervals']
    df_left = df_left[df_left.columns[1:]]  # discarding first bin
    df_cent = df['2DS10']
    df_cent.columns = df_cent.attrs['2DS10']['intervals']
    df_cent = df_cent[df_cent.columns[1:]]  # discarding first bin
    df_right = df['HVPS']
    df_right.columns = df_right.attrs['HVPS']['intervals']
    df_right = df_right[df_right.columns[1:]]  # discarding first bin

    _left = df_left.iloc[:, df_left.columns.overlaps(_intervals[0])]
    _cent_left = df_cent.iloc[:, df_cent.columns.overlaps(_intervals[0])]
    _cent_right = df_cent.iloc[:, df_cent.columns.overlaps(_intervals[-1])]
    _right = df_right.iloc[:, df_right.columns.overlaps(_intervals[-1])]
    instr = ['FCDP', '2DS10_L', '2DS10_R', 'HVPS']

    nd_overlap = pd.concat([_left.reindex(columns=np.arange(_left.columns.min().mid,
                                                            _left.columns.max().right, 0.5)),
                            _cent_left.reindex(columns=np.arange(_cent_left.columns.min().mid,
                                                                 _cent_left.columns.max().right, 0.5)),
                            _cent_right.reindex(columns=np.arange(_cent_right.columns.min().right,
                                                                  _cent_right.columns.max().right, 1)),
                            _right.reindex(columns=np.arange(_right.columns.min().mid,
                                                             _right.columns.max().mid, 1))],
                           axis=1, keys=instr, levels=[instr])

    if method == 'linear':
        _fcdp = nd_overlap['FCDP'] * (_intervals[0].right - nd_overlap['FCDP'].columns) / \
                (_intervals[0].right - _intervals[0].left)

        _2ds_l = nd_overlap['2DS10_L'] * (nd_overlap['2DS10_L'].columns - _intervals[0].left) / \
                 (_intervals[0].right - _intervals[0].left)

        _2ds_r = nd_overlap['2DS10_R'] * (_intervals[-1].right - nd_overlap['2DS10_R'].columns) / \
                 (_intervals[-1].right - _intervals[-1].left)

        _hvsp = nd_overlap['HVPS'] * (nd_overlap['HVPS'].columns - _intervals[-1].left) / \
                (_intervals[-1].right - _intervals[-1].left)

        res = pd.concat([_fcdp, _2ds_l, _2ds_r, _hvsp], axis=1, keys=['fcdp', '2ds_l', '2ds_s', 'hvps'],
                        levels=[['fcdp', '2ds_l', '2ds_s', 'hvps']]).stack().sum(axis=1).unstack()

        res = res.reindex(columns=df_left.columns[df_left.columns.overlaps(_intervals[0])].append(
            df_cent.columns[df_cent.columns.overlaps(_intervals[-1])]).mid)
        res.columns = df_left.columns[df_left.columns.overlaps(_intervals[0])].append(
            df_cent.columns[df_cent.columns.overlaps(_intervals[-1])])
        nd_res = pd.concat([df_left.iloc[:, ~df_left.columns.overlaps(_intervals[0])],
                            res.iloc[:, res.columns.overlaps(_intervals[0])],
                            df_cent.iloc[:, df_cent.columns.overlaps(_intervals[1])],
                            res.iloc[:, res.columns.overlaps(_intervals[-1])],
                            df_right.iloc[:, ~df_right.columns.overlaps(_intervals[-1])]],
                           axis=1)

        nd_res = nd_res[sorted(nd_res.columns)]
        nd_res.attrs['intervals'] = nd_res.columns
        d_d = {i.mid: i.length for i in nd_res.columns}
        nd_res.columns = nd_res.columns.mid
        nd_res.attrs['dsizes'] = d_d
        return nd_res

    elif 'snal':
        _fcdp_2ds_0 = nd_overlap.stack()[(((nd_overlap.stack()['FCDP'] == 0) &
                                           (nd_overlap.stack()['FCDP'].notnull())) &
                                          ((nd_overlap.stack()['2DS10_L'] == 0) &
                                           (nd_overlap.stack()['2DS10_L'].notnull())))]['FCDP'].unstack()

        _fcdp_2ds_null = nd_overlap.stack()[(((nd_overlap.stack()['FCDP'] == 0) &
                                              (nd_overlap.stack()['FCDP'].notnull())) &
                                             (nd_overlap.stack()['2DS10_L'].isnull()))]['FCDP'].unstack()

        _2ds_fcdp_null = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_L'] == 0) &
                                              (nd_overlap.stack()['2DS10_L'].notnull())) &
                                             (nd_overlap.stack()['FCDP'].isnull()))]['2DS10_L'].unstack()

        _fcdp_2ds_fdcp = nd_overlap.stack()[(((nd_overlap.stack()['FCDP'] != 0) &
                                              (nd_overlap.stack()['FCDP'].notnull())) &
                                             ((nd_overlap.stack()['2DS10_L'] == 0) |
                                              (nd_overlap.stack()['2DS10_L'].isnull())))]['FCDP'].unstack()

        _2ds_fcdp_2ds = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_L'] != 0) &
                                             (nd_overlap.stack()['2DS10_L'].notnull())) &
                                            ((nd_overlap.stack()['FCDP'] == 0) |
                                             (nd_overlap.stack()['FCDP'].isnull())))]['2DS10_L'].unstack()

        _sfc_wt = nd_overlap.stack()[(((nd_overlap.stack()['FCDP'] != 0) &
                                       (nd_overlap.stack()['FCDP'].notnull())) &
                                      ((nd_overlap.stack()['2DS10_L'] != 0) & (
                                          nd_overlap.stack()['2DS10_L'].notnull())))]

        _sfc_wt['wt_fcdp'] = _sfc_wt['FCDP'] * (_intervals[0].right - _sfc_wt.index.get_level_values(1)) / \
                             (_intervals[0].right - _intervals[0].left)

        _sfc_wt['wt_2ds'] = _sfc_wt['2DS10_L'] * (_sfc_wt.index.get_level_values(1) - _intervals[0].left) / \
                            (_intervals[0].right - _intervals[0].left)

        _fcd_2ds = _sfc_wt[['wt_fcdp', 'wt_2ds']].sum(axis=1).unstack()

        _res_left = pd.concat([_fcdp_2ds_0, _fcdp_2ds_null, _2ds_fcdp_null, _fcdp_2ds_fdcp, _2ds_fcdp_2ds, _fcd_2ds],
                              axis=1, levels=[['_fcdp_2ds_0', '_fcdp_2ds_null', '_2ds_fcdp_null', '_fcdp_2ds_fdcp',
                                               '_2ds_fcdp_2ds', '_fcd_2ds']],
                              keys=['_fcdp_2ds_0', '_fcdp_2ds_null', '_2ds_fcdp_null', '_fcdp_2ds_fdcp',
                                    '_2ds_fcdp_2ds', '_fcd_2ds']).stack().sum(axis=1).unstack()
        _res_left = _res_left.reindex(columns=df_left.columns[df_left.columns.overlaps(_intervals[0])].mid)
        _res_left.columns = df_left.columns[df_left.columns.overlaps(_intervals[0])]
        _2ds_hvps_0 = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_R'] == 0) &
                                           (nd_overlap.stack()['2DS10_R'].notnull())) &
                                          ((nd_overlap.stack()['HVPS'] == 0) &
                                           (nd_overlap.stack()['HVPS'].notnull())))]['2DS10_R'].unstack()

        _2ds_hvps_null = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_R'] == 0) &
                                              (nd_overlap.stack()['2DS10_R'].notnull())) &
                                             (nd_overlap.stack()['HVPS'].isnull()))]['2DS10_R'].unstack()

        _hvps_2ds_null = nd_overlap.stack()[(((nd_overlap.stack()['HVPS'] == 0) &
                                              (nd_overlap.stack()['HVPS'].notnull())) &
                                             (nd_overlap.stack()['2DS10_R'].isnull()))]['HVPS'].unstack()

        _2ds_hvps_2ds = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_R'] != 0) &
                                             (nd_overlap.stack()['2DS10_R'].notnull())) &
                                            ((nd_overlap.stack()['HVPS'] == 0) |
                                             (nd_overlap.stack()['HVPS'].isnull())))]['2DS10_R'].unstack()

        _hvps_2ds_hvps = nd_overlap.stack()[(((nd_overlap.stack()['HVPS'] != 0) &
                                              (nd_overlap.stack()['HVPS'].notnull())) &
                                             ((nd_overlap.stack()['2DS10_R'] == 0) |
                                              (nd_overlap.stack()['2DS10_R'].isnull())))]['HVPS'].unstack()

        _sdd = nd_overlap.stack()[(((nd_overlap.stack()['2DS10_R'] != 0) &
                                    (nd_overlap.stack()['2DS10_R'].notnull())) &
                                   ((nd_overlap.stack()['HVPS'] != 0) &
                                    (nd_overlap.stack()['HVPS'].notnull())))]

        _sdd['2ds10_wgt'] = _sdd['2DS10_R'] * (_intervals[-1].right - _sdd.index.get_level_values(1)) / \
                            (_intervals[-1].right - _intervals[1].left)
        _sdd['hvps_wgt'] = _sdd['HVPS'] * (_sdd.index.get_level_values(1) - _intervals[-1].left) / \
                           (_intervals[-1].right - _intervals[-1].left)
        _sdd['nd_res'] = _sdd[['2ds10_wgt', 'hvps_wgt']].sum(1)

        _sdd = _sdd['nd_res'].unstack()

        res = pd.concat([_2ds_hvps_0, _hvps_2ds_null, _2ds_hvps_null, _2ds_hvps_2ds, _hvps_2ds_hvps, _sdd],
                        axis=1, levels=[['_2ds_hvps_0', '_hvps_2ds_null', '_2ds_hvps_null', '_2ds_hvps_2ds',
                                         '_hvps_2ds_hvps', '_sdd']],
                        keys=['_2ds_hvps_0', '_hvps_2ds_null', '_2ds_hvps_null', '_2ds_hvps_2ds',
                              '_hvps_2ds_hvps', '_sdd']).stack().sum(1).unstack()

        res = res.reindex(columns=df_cent.columns[df_cent.columns.overlaps(_intervals[-1])].mid)
        res.columns = df_cent.columns[df_cent.columns.overlaps(_intervals[-1])]

        nd_res = pd.concat([df_left.iloc[:, ~df_left.columns.overlaps(_intervals[0])],
                            _res_left,
                            df_cent.iloc[:, df_cent.columns.overlaps(_intervals[1])],
                            res,
                            df_right.iloc[:, ~df_right.columns.overlaps(_intervals[-1])]], axis=1)
        nd_res = nd_res.iloc[:, ~nd_res.columns.duplicated()]
        d_d = {i.mid: i.length for i in nd_res.columns}
        nd_res.columns = nd_res.columns.mid
        nd_res.attrs['dsizes'] = d_d
        nd_res.attrs['instrument'] = 'Composite_PSD'
        return nd_res


def vel(d):
    return -0.1021 + 4.932 * d - 0.955 * d ** 2 + 0.07934 * d ** 3 - 0.0023626 * d ** 4


def pds_parameters(nd):
    """
    Compute the psd parameters
    :param nd: partice size distribution in # L-1 um-1
    :return: list with lwc, dm, nw, z, and r
    """
    d = np.fromiter(nd.columns, dtype=float) / 1e3  # diameter in millimeters
    d_d = np.fromiter(nd.attrs['dsizes'].values(), dtype=float) / 1e3  # d_size in um
    lwc = nd.mul(1e6).mul(d ** 3).mul(d_d) * (np.pi / (6 * 1000))  # g / m3
    dm = nd.mul(d ** 4).mul(d_d).sum(1) / nd.mul(d ** 3).mul(d_d).sum(1)  # mm
    nw = 1e3 * (4 ** 4 / np.pi) * (lwc.sum(1) / dm ** 4)
    z = nd.mul(d ** 6).mul(d_d)
    _ = ['lwc', 'dm', 'nw', 'z']
    return pd.concat([lwc, dm, nw, z], axis=1, keys=_, levels=[_])


def bcksct(ds, instrument, _lower, _upper, ar=1, j=0) -> pd.DataFrame:
    """

    :param _upper: upper diameter of the pds
    :param _lower: upper diameter of the pds
    :param instrument:
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
    path_db = f'{path_data}/cloud_probes/db'
    make_dir(path_db)
    str_db = f"sqlite:///{path_db}/backscatter_{_lower}_{_upper}.sqlite"
    df_scatter.to_sql(f'{instrument}', con=str_db, if_exists='replace')
    return df_scatter


def ref_calc(nd, _lower, _upper, mie=False):
    ds = np.fromiter(nd.attrs['dsizes'].keys(), dtype=float) / 1e3
    try:
        path_db = f'{path_data}/cloud_probes/db'
        str_db = f"sqlite:///{path_db}/backscatter_{_lower}_{_upper}.sqlite"
        backscatter = pd.read_sql(f"{nd.attrs['instrument']}", con=str_db)
    except (OperationalError, ValueError):
        backscatter = bcksct(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper)

    if len(ds) != backscatter.shape[0]:
        backscatter = bcksct(ds, nd.attrs['instrument'], _lower=_lower, _upper=_upper)

    dsizes = np.fromiter(nd.attrs['dsizes'].values(), dtype=float) / 1e3
    ku_wvl = c / 14e9 * 1000
    ka_wvl = c / 35e9 * 1000
    w_wvl = c / 95e9 * 1000
    wvl = ['z_Ku', 'z_Ka', 'z_W']
    if mie:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_Ku'].values,
                                                                     axis='columns').mul(dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_Ka'].values,
                                                                     axis='columns').mul(dsizes)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['Mie_W'].values,
                                                                   axis='columns').mul(dsizes)
        df_z = pd.concat([z_ku, z_ka, z_w], axis=1, keys=wvl, levels=[wvl])
        return df_z
    else:
        z_ku = (ku_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_Ku'].values,
                                                                     axis='columns').mul(dsizes)
        z_ka = (ka_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_Ka'].values,
                                                                     axis='columns').mul(dsizes)
        z_w = (w_wvl ** 4 / (np.pi ** 5 * 0.93)) * nd.mul(1e6).mul(backscatter['T_mat_W'].values,
                                                                   axis='columns').mul(dsizes)
        df_z = pd.concat([z_ku, z_ka, z_w], axis=1, keys=wvl, levels=[wvl])
        return df_z


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
        df_add = df_add.filter(['Temp', 'Dew', 'Palt', 'NevLWC', 'VaV'])
        df_add['RH'] = compute_hr(t=df_add['Temp'], td=df_add['Dew'])
        df_add['RH'] = df_add['RH'].where(df_add['RH'] <= 100, 100)
        cols = ['temp', 'dew_point', 'altitude', 'lwc', 'vertical_vel', 'RH']
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
                                ' Vertical_Speed_YANG_MetNav', ' Relative_Humidity_YANG_MetNav'])
        cols = ['temp', 'dew_point', 'altitude', 'lwc', 'vertical_vel', 'RH']
        new_cols = {j: cols[i] for i, j in enumerate(list(df_add.columns))}
        df_add.rename(columns=new_cols, inplace=True)
        return df_add


def area_filter(ds):
    diameter = ds.attrs['bin_cent'] / 1e3  # mm
    ar_func = lambda \
        d: 0.9951 + 0.0251 * d - 0.03644 * d ** 2 + 0.005303 * d ** 3 - 0.0002492 * d ** 4  # diameter in mm
    ar = ar_func(diameter)  # mm
    area_func = lambda x: np.pi * (x / 2) ** 2
    area = area_func(diameter) / 1e5
    _lower = area * ar * 0.5
    _upper = area * ar * 2
    df_area = ds.filter(like='a_bin')
    df_cnt = ds.filter(like='cnt')
    df_filter = pd.DataFrame(df_area.values / df_cnt.values, index=ds.index, columns=ds.filter(like='nsd').columns)
    ds_area = df_filter[(df_filter >= _lower) & (df_filter <= _upper)].notnull()
    cols = ds.filter(like='nsd').columns
    ds.loc[:, cols] = ds.filter(like='nsd')[ds_area]
    return ds


def fill_2ds(ls_df):
    df_2ds = [i for i in ls_df if i.attrs['instrument'] == '2DS10'][0]
    df_h2ds = [i for i in ls_df if i.attrs['instrument'] == 'Hawk2DS10'][0]
    df_2ds = df_2ds.fillna(df_h2ds)
    ls_df = [df_2ds if i.attrs['instrument'] == '2DS10' else i for i in ls_df]
    return ls_df


def main():
    aircraft = ['Lear', 'P3B']
    for air in aircraft:
        intervals = [300, 1000]
        _lower = intervals[0]
        _upper = intervals[-1]
        ls_df = get_data(air, temp=2, sensors=['2DS10', 'HVPS', 'Hawk2DS10'])
        ls_df = fill_2ds(ls_df)
        hvps = area_filter([i for i in ls_df if i.attrs['instrument'] == 'HVPS'][0])
        ls_df = [hvps if i.attrs['instrument'] == 'HVPS' else i for i in ls_df]
        ls_df = filter_by_cols(ls_df)
        instr = [i.attrs['instrument'] for i in ls_df]
        attrs = [i.attrs for i in ls_df]
        dt_attrs = {instr[i]: j for i, j in enumerate(attrs)}
        for idx, att in enumerate(attrs):
            ls_df[idx].attrs = attrs[0]
        df_concat = pd.concat(ls_df[:1], axis=1, keys=instr, levels=[instr])
        df_concat.attrs = dt_attrs

        if location in ['atmos', 'alfonso']:
            if air == "Lear":
                indexx = pd.date_range(start='2019-09-07 2:31:45', periods=150, tz='UTC', freq='S')  # for Lear
            else:
                indexx = pd.date_range(start='2019-08-27 00:15', periods=120, tz='UTC', freq='S')  # for P3B
        else:
            indexx = df_concat.index

        df_concat = df_concat[(df_concat.index >= f"{indexx.min()}") & (df_concat.index <= f"{indexx.max()}")]
        df_merged = linear_wgt(df_concat['2DS10'], df_concat['HVPS'], ovr_upp=intervals[-1], ovr_lower=intervals[0],
                               method='snal')
        df_reflectivity = ref_calc(df_merged, _upper=_upper, _lower=_lower)
        params = pds_parameters(df_merged)
        df_add = get_add_data(air, indexx=indexx)
        d_d = np.fromiter(df_merged.attrs['dsizes'].values(), dtype=float)
        df_merged = df_merged.join(df_add)
        xr_merg = xr.Dataset(
            data_vars=dict(
                psd=(["time", "diameter"], df_merged[df_merged.columns[:-6]].to_numpy()),
                refl_ku=(["time", "diameter"], df_reflectivity['z_Ku'].to_numpy()),
                refl_ka=(["time", "diameter"], df_reflectivity['z_Ka'].to_numpy()),
                refl_w=(["time", "diameter"], df_reflectivity['z_W'].to_numpy()),
                lwc=(["time", "diameter"], params['lwc'].to_numpy()),
                nw=(["time"], params['nw'].to_numpy()[:, 0]),
                dm=(["time"], params['dm'].to_numpy()[:, 0]),
                z=(["time", "diameter"], params['z'].to_numpy()),
                # r=(["time"], params['r'].to_numpy()),
                temp=(["time"], df_merged['temp'].to_numpy()),
                dew_point=(["time"], df_merged['dew_point'].to_numpy()),
                altitude=(["time"], df_merged['altitude'].to_numpy()),
                lwc_plane=(["time"], df_merged['lwc'].to_numpy()),
                vert_vel=(["time"], df_merged['vertical_vel'].to_numpy()),
                RH=(["time"], df_merged['RH'].to_numpy()),
                d_d=(["diameter"], d_d)
            ),
            coords=dict(
                time=(["time"], np.array([i.to_datetime64() for i in df_merged.index])),
                diameter=(["diameter"], df_merged.columns[:-6])),
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
        xr_merg = xr_merg.sel(diameter=slice(20, 3824.5))
        store = f"{path_data}/cloud_probes/zarr/combined_psd_{air}_{_lower}_{_upper}.zarr"
        xr_merg.to_zarr(store=store, consolidated=True)
        print(1)


if __name__ == '__main__':
    main()
