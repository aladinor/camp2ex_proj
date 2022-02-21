#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import pandas as pd
import sys
import os
import numpy as np
from re import split, findall
import xarray as xr
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini


def moment_nth(sr_nd, dict_diameters, moment):
    mn = sr_nd * dict_diameters['delta_diam'] * dict_diameters['diameters'] ** moment
    return mn.sum()


class Ict2df(object):
    def __init__(self, _file):
        self.path_file = _file
        self.dt, self.dt_sizes, self.sizes, self.header, self.file_type, self.bin_cent = self._get_meta()
        self.instrument = self.path_file.split('/')[-1].split('-')[-1].split('_')[0]
        self.aircraft = self.path_file.split('/')[-1].split('-')[-1].split('_')[1]
        self.df = self._read_file()

    def _get_meta(self):
        with open(self.path_file, 'r') as f:
            lines = f.readlines()
            header, file_type = findall(r"\d*\.\d+|\d+", lines[0])
            sizes = np.array([float(''.join(findall(r"\d*\.\d+|\d+", i[i.find('(') + 1: i.find(')')])[:1]))
                              for i in lines[:200] if i.startswith('cbin')])
            dsizes = sizes[1:] - sizes[:-1]
            dsizes = np.append(dsizes, dsizes[-1])
            bin_cent = (sizes[1:] - sizes[:-1]) / 2 + sizes[:-1]
            bin_cent = np.append(bin_cent, sizes[-1] + dsizes[-1])
            dt_sizes = {i: j for i, j in zip(bin_cent, dsizes)}
            try:
                dt = pd.to_datetime(''.join(lines[6].split(',')[:3]), format='%Y%m%d', utc=True)
                return dt, dt_sizes, sizes, int(header) - 1, file_type, bin_cent
            except ValueError:
                dt = pd.to_datetime(''.join(lines[6].split(',')[:3]).replace(' ', ''), format='%Y%m%d', utc=True)
                return dt, dt_sizes, sizes, int(header) - 1, file_type, bin_cent

    def _read_file(self):
        df = pd.read_csv(self.path_file, skiprows=self.header, header=0, na_values=[-999, -9.99])
        df['time'] = df.Time_Start.map(lambda x: self.dt + pd.to_timedelta(x, unit='seconds'))
        df['time'] = df['time'].map(lambda x: x.to_datetime64())
        df.index = df['time']
        df.attrs = {'sizes': self.sizes, 'dsizes': self.dt_sizes, 'bin_cent': self.bin_cent}
        cols = df.filter(like='cbin', axis=1).columns.tolist()
        names = [f'nsd {self.sizes[i]}-{self.sizes[i+1]}' for i, j in enumerate(self.sizes[:-1])]
        names.append(f'>{self.sizes[-1]}')
        dt_cols = {j: names[i] for i, j in enumerate(cols)}
        df = df.rename(columns=dt_cols)
        return df


def pd2xr(df_concat, example):
    ds = xr.Dataset.from_dataframe(df_concat)
    ds.attrs['dt_sizes'] = example.dt_sizes
    ds.attrs['ls_sizes'] = example.sizes
    return ds


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']

    # Data
    # FCDP - Fast Cloud Droplet Probe (0.0 - 50 um)
    _file1 = f'{path_data}/data/LAWSON_PAUL/FCDP/CAMP2Ex-FCDP_Learjet_20190907_R1_L1.ict'
    _file2 = f'{path_data}/data/LAWSON_PAUL/FCDP/CAMP2Ex-FCDP_P3B_20190824_R1.ict'

    # FFSSP - Fast Forward Scattering Spectrometer Probe (0.0 - 50 um)
    _file3 = f'{path_data}/data/LAWSON_PAUL/Hawk2DS10/CAMP2Ex-FFSSP_Learjet_20190907_R0_L1.ict'

    # 2DS10 - Optical Array Spectrometer (10um - 3mm)
    _file4 = f'{path_data}/data/LAWSON_PAUL/HVPS/CAMP2Ex-2DS10_Learjet_20190907_R0_L1.ict'
    _file5 = f'{path_data}/data/LAWSON_PAUL/HVPS/CAMP2Ex-2DS10_P3B_20190925_R0.ict'

    # HVPS - High Volume Precipitation Spectrometer (150um-2cm)
    _file6 = f'{path_data}/data/LAWSON_PAUL/HVPS/CAMP2Ex-HVPS_Learjet_20190907_R0_L1.ICT'
    _file7 = f'{path_data}/data/LAWSON_PAUL/HVPS/CAMP2Ex-HVPS_P3B_20190915_R0.ict'

    # Hawk2DS10 - Optical Array Spectrometer on the Hawk instrument (10um - 3mm)
    _file8 = f'{path_data}/data/LAWSON_PAUL/Hawk2DS10/CAMP2Ex-Hawk2DS10_Learjet_20190907_R0_L1.ict'
    _file9 = f'{path_data}/data/LAWSON_PAUL/Hawk2DS10/CAMP2Ex-Hawk2DS10_P3B_20190923_R0.ict'

    # Hawk2DS50 - Optical Array Spectrometer on the Hawk instrument (10um - 3mm)
    _file10 = f'{path_data}/data/LAWSON_PAUL/Hawk2DS50/CAMP2Ex-Hawk2DS50_Learjet_20190907_R0_L1.ict'
    _file11 = f'{path_data}/data/LAWSON_PAUL/Hawk2DS50/CAMP2Ex-Hawk2DS50_P3B_20190927_R0.ict'

    # HawkFCDP - Fast Cloud Droplet Probe on the Hawk intrument (0.0 - 50 um)
    _file12 = f'{path_data}/data/LAWSON_PAUL/HawkFCDP/CAMP2Ex-HawkFCDP_P3B_20191005_R1.ict'
    _file13 = f'{path_data}/data/LAWSON_PAUL/HawkFCDP/CAMP2Ex-HawkFCDP_Learjet_20190913_R1.ict'

    files = [_file1, _file2, _file3, _file4, _file5, _file6, _file7, _file8, _file9, _file10, _file11, _file12, _file13]
    for _file in files:
        _type = _file.split('/')[-1].split('-')[-1].split('_')[0]
        _aircraft = _file.split('/')[-1].split('-')[-1].split('_')[1]
        files = glob.glob(f'{path_data}/data/LAWSON_PAUL/{_type}/CAMP2Ex-{_type}_{_aircraft}*')
        ls_pd = [Ict2df(i).df for i in files]
        attrs = ls_pd[0].attrs
        df_all = pd.concat(ls_pd)
        df_all.attrs = attrs
        df_all.to_pickle(f'{path_data}/data/LAWSON_PAUL/{_type}/{_type}_{_aircraft}.pkl')
    pass


if __name__ == '__main__':
    main()
