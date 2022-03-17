#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from typing import List

import pandas as pd
import sys
import os
import numpy as np
from re import split, findall
import xarray as xr
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir


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
            lines: list[str] = f.readlines()
            header, file_type = findall(r"\d*\.\d+|\d+", lines[0])
            try:
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
            except IndexError:
                try:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]), format='%Y%m%d', utc=True)
                    return dt, None, None, int(header) - 1, file_type, None
                except ValueError:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]).replace(' ', ''), format='%Y%m%d', utc=True)
                    return dt, None, None, int(header) - 1, file_type, None
            pass

    def _read_file(self):
        df = pd.read_csv(self.path_file, skiprows=self.header, header=0, na_values=[-999, -9.99])
        df['time'] = df.Time_Start.map(lambda x: self.dt + pd.to_timedelta(x, unit='seconds'))
        df['time'] = df['time'].map(lambda x: x.to_datetime64())
        df.index = df['time'].dt.tz_localize('utc')
        df['local_time'] = df['time'].dt.tz_localize('utc').dt.tz_convert('Asia/Manila')
        df.drop(columns=['time'], axis=1, inplace=True)
        df.attrs = {'sizes': self.sizes, 'dsizes': self.dt_sizes, 'bin_cent': self.bin_cent, 'aircraft': self.aircraft,
                    'instrument': self.instrument}
        try:
            cols = df.filter(like='cbin', axis=1).columns.tolist()
            names = [f'nsd {self.sizes[i]}-{self.sizes[i + 1]}' for i, j in enumerate(self.sizes[:-1])]
            names.append(f'nsd >{self.sizes[-1]}')
            if (self.aircraft == 'P3B') and (self.instrument == 'Hawk2DS50'):
                names = ['nsd 3025.0-3250.0' if (item == 'nsd 3025.0-3225.0') or
                         (item == 'nsd 3025.0-3275.0') else item for item in names]
                names = ['nsd 3250.0-3525.0' if (item == 'nsd 3225.0-3525.0') or
                         (item == 'nsd 3275.0-3525.0') else item for item in names]
            dt_cols = {j: names[i] for i, j in enumerate(cols)}
            df = df.rename(columns=dt_cols)
            return df
        except TypeError:
            return df


def ict2pkl(files, path_save):
    try:
        _file = files[0]
        _type = _file.split('/')[-1].split('-')[-1].split('_')[0]
        _aircraft = _file.split('/')[-1].split('-')[-1].split('_')[1]
        ls_pd = [Ict2df(i).df for i in files]
        attrs = ls_pd[0].attrs
        attrs['type'] = _type
        attrs['aircraft'] = _aircraft
        df_all = pd.concat(ls_pd)
        df_all.attrs = attrs
        df_all = df_all.sort_index()
        path = f'{path_save}/{_aircraft.upper()}/all'
        make_dir(path)
        df_all.to_pickle(f'{path}/{_type}_{_aircraft}.pkl')
    except IndexError:
        pass


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']

    instruments = ['FCDP', '2DS10', 'HVPS', 'FFSSP', 'Hawk2DS10', 'Hawk2DS50', 'HawkFCDP', 'Page0']
    aircraft = ['P3B', 'Learjet']
    file_type = [f'{path_data}/data/LAWSON.PAUL/{i.upper()}/{j}/CAMP2Ex-{j}_{i}_' for i in aircraft for j in instruments]
    path_save = f'{path_data}/data/LAWSON.PAUL'
    for file in file_type:
        files = glob.glob(f'{file}*')
        if files:
            ict2pkl(files, path_save)

    # files = glob.glob(f'{path_data}/data/01_SECOND.P3B_MRG/MERGE/p3b/*.ict')
    # path_save = f'{path_data}/data/01_SECOND.P3B_MRG'
    # ict2pkl(files, path_save=path_save)


if __name__ == '__main__':
    main()
