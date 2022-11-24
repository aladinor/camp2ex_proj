#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import pandas as pd
import sys
import os
import numpy as np
import xarray as xr
from zarr.errors import ContainsGroupError
from re import split, findall
from itertools import chain

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


class Ict2df(object):
    def __init__(self, _file):
        self.path_file = _file
        self.dt, self.dt_sizes, self.sizes, self.header, self.file_type, self.bin_cent, self.intervals, \
        self.na_vals, self.units = self._get_meta()
        self.instrument = self.path_file.split('/')[-1].split('-')[-1].split('_')[0]
        self.aircraft = self.path_file.split('/')[-1].split('-')[-1].split('_')[1]
        self.df = self._read_file()

    def _get_meta(self):
        with open(self.path_file, 'r') as f:
            lines: list[str] = f.readlines()
            header, file_type = findall(r"\d*\.\d+|\d+", lines[0])
            units = list(set([i.replace("'", "").split(",")[1] for i in lines if i.startswith('cbin')]))
            try:
                intervals = [list(map(float, findall(r"\d*\.\d+|\d+", lines[i])[1:]))
                             for i in range(len(lines)) if lines[i].startswith('cbin')]
                intervals = [pd.Interval(*i) for i in intervals[:-1]]
                intervals.append(intervals[-1] + intervals[-1].length)
                sizes = np.array([i.left for i in intervals])
                dsizes = np.array([i.length for i in intervals])
                bin_cent = np.array([i.mid for i in intervals])
                na_val = np.array(list(set([float(i) for i in lines[11].replace("'", "").split(',')])), dtype='float64')
                try:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]), format='%Y%m%d', utc=True)
                    return dt, dsizes, sizes, int(header) - 1, file_type, bin_cent, intervals, na_val, units
                except ValueError:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]).replace(' ', ''), format='%Y%m%d', utc=True)
                    return dt, dsizes, sizes, int(header) - 1, file_type, bin_cent, intervals, na_val, units
            except IndexError:
                try:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]), format='%Y%m%d', utc=True)
                    return dt, None, None, int(header) - 1, file_type, None, None, None, None
                except ValueError:
                    dt = pd.to_datetime(''.join(lines[6].split(',')[:3]).replace(' ', ''), format='%Y%m%d', utc=True)
                    return dt, None, None, int(header) - 1, file_type, None, None, None, None
            pass

    def _read_file(self):
        df = pd.read_csv(self.path_file, skiprows=self.header, header=0, na_values=self.na_vals)
        df['time'] = df['Time_Start'].map(lambda x: self.dt + pd.to_timedelta(x, unit='seconds'))
        df.index = df['time']
        df['local_time'] = df['time'].dt.tz_convert('Asia/Manila')
        df.drop(columns=['time'], axis=1, inplace=True)
        df.attrs = {'sizes': self.sizes, 'dsizes': self.dt_sizes, 'bin_cent': self.bin_cent, 'aircraft': self.aircraft,
                    'instrument': self.instrument, 'intervals': self.intervals, 'psd_units': self.units}
        try:
            _cols = ['cbin', 'nbin', 'mbin', 'abin']
            names = ['nsd', 'cnt', 'm_bin', 'a_bin']
            cols = list(chain.from_iterable([df.filter(like=i, axis=1).columns.tolist() for i in _cols]))
            dt_cols = self.change_cols_name(cols, names)
            df = df.rename(columns=dt_cols)
            return df
        except TypeError:
            return df

    def change_cols_name(self, cols, names):
        _nam = []
        for name in names:
            _names = [[f'{name} {i.left}-{i.right}' for i in self.intervals[:-1]],
                      [f'{name} >{self.intervals[-1].left}']]
            _names = list(chain.from_iterable(_names))
            if (self.aircraft == 'P3B') and (self.instrument == 'Hawk2DS50'):
                _names = [f'{name} 3025.0-3250.0' if (item == f'{name} 3025.0-3225.0') or
                                                     (item == f'{name} 3025.0-3275.0') else item for item in _names]
                _names = [f'{name} 3250.0-3525.0' if (item == f'{name} 3225.0-3525.0') or
                                                     (item == f'{name} 3275.0-3525.0') else item for item in _names]
            _nam.append(_names)
        _nam = list(chain.from_iterable(_nam))
        dt = {j: _nam[i] for i, j in enumerate(cols)}
        return dt


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

        # pickle
        path_pk = f'{path_data}/pkl'
        make_dir(path_pk)
        df_all.to_pickle(f'{path_pk}/{_type}_{_aircraft}.pkl')

        # zarr
        nsd = df_all.filter(like='nsd').columns.to_list()
        cnt = df_all.filter(like='cnt').columns.to_list()
        mbn = df_all.filter(like='m_bin').columns.to_list()
        abn = df_all.filter(like='a_bin').columns.to_list()
        other = [i for i in df_all.columns.to_list() if not (i.startswith('nsd') | i.startswith('Time') |
                                                             i.startswith('local') | i.startswith('cnt') |
                                                             i.startswith('a_bin') | i.startswith('m_bin'))]
        psd_dict = {'psd': (["time", "diameter"], df_all[nsd].to_numpy())}
        cnt_dict = {'cnt_bin': (["time", "diameter"], df_all[cnt].to_numpy())}
        if mbn:
            mbn_dict = {'m_bin': (["time", "diameter"], df_all[mbn].to_numpy())}
        else:
            mbn_dict = {}
        if abn:
            abn_dict = {'a_bin': (["time", "diameter"], df_all[abn].to_numpy())}
        else:
            abn_dict = {}
        other_dict = {i: (["time"], df_all[i].to_numpy()) for i in other}
        d_d = {'d_d': (["diameter"], df_all.attrs['dsizes'])}
        local_t = {'local_time': (["time"], np.array([i.to_datetime64() for i in df_all["local_time"]]))}
        attrs = df_all.attrs

        if (df_all.attrs['instrument'] == 'p3b') & (df_all.attrs['aircraft'] == 'merge'):
            data = other_dict | local_t
            coords = dict(time=(["time"], np.array([i.to_datetime64() for i in df_all.index])))
            for key, value in dict(attrs).items():
                if value is None:
                    del attrs[key]
        elif (df_all.attrs['instrument'] == 'Page0') & (df_all.attrs['aircraft'] == 'Learjet'):
            data = other_dict | local_t
            coords = dict(time=(["time"], np.array([i.to_datetime64() for i in df_all.index])))
            for key, value in dict(attrs).items():
                if value is None:
                    del attrs[key]
        else:
            data = psd_dict | cnt_dict | mbn_dict | abn_dict | other_dict | local_t | d_d
            coords = dict(time=(["time"], np.array([i.to_datetime64() for i in df_all.index])),
                          diameter=(["diameter"], df_all.attrs['bin_cent']))
            del attrs['intervals']

        print(f"{df_all.attrs['instrument']}_{df_all.attrs['aircraft']}.zarr")

        xr_data = xr.Dataset(
            data_vars=data,
            coords=coords,
            attrs=attrs
        )
        path_zarr = f'{path_data}/zarr'
        make_dir(path_zarr)
        store = f"{path_zarr}/{df_all.attrs['instrument']}_{df_all.attrs['aircraft']}.zarr"
        try:
            _ = xr_data.to_zarr(store=store, consolidated=True)
        except ContainsGroupError:
            print(f"{df_all.attrs['instrument']}_{df_all.attrs['aircraft']}.zarr already exist. Delete it first!")
        del df_all
    except IndexError:
        pass


def main():

    instruments = ['FCDP', '2DS10', 'HVPS', 'FFSSP', 'Hawk2DS10', 'Hawk2DS50', 'HawkFCDP', 'Page0']
    aircraft = ['Learjet', 'P3B']
    file_type = [f'{path_data}/data/LAWSON.PAUL/{i.upper()}/{j}' for i in aircraft for j in instruments]
    path_save = f'{path_data}/data/LAWSON.PAUL'
    for file in file_type:
        files = glob.glob(f'{file}/*.ict')
        if not files:
            _unit = file.split(':')[0]
            files = glob.glob(f"/mnt/{file.split(':')[0].lower()}/{file.split(':')[-1]}/*.ict")
        if files:
            ict2pkl(files, path_save)

    files = glob.glob(f'{path_data}/data/01_SECOND.P3B_MRG/MERGE/p3b/*.ict')
    path_save = f'{path_data}/data/01_SECOND.P3B_MRG'
    ict2pkl(files, path_save=path_save)


if __name__ == '__main__':
    main()
