#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import h5py
import xarray as xr
import numpy as np
from utils import get_pars_from_ini
from datetime import datetime


def get_time(time_array, numbers):
    """
    Functions that creates a 3d time array from timestamps
    :param time_array: 2d timestamp array
    :param numbers: number of times in the new axis
    :return: 3d time array
    """
    v_func = np.vectorize(lambda x: datetime.fromtimestamp(x))
    _time = v_func(time_array)
    time_3d = np.repeat(_time[np.newaxis, :, :], numbers, axis=0)
    return time_3d


def hdf2xr(h5_path):
    """
    Function that converts CAMP2EX files (hdf5 files) to xarray datasets
    :param h5_path: full path to the hdf5 file
    :return: a dictionary with groups a key and datasets as values
    """
    dt_params = get_pars_from_ini()
    h5f = h5py.File(h5_path, mode='r')
    groups = [i[0] for i in h5f.items()]
    members = {i: [j[0] for j in h5f.get(i).items()] for i in groups}
    ds_res = {}
    for group in groups[3:4]:
        ds = xr.Dataset()
        for key in members[group]:
            if h5f[group][key].size == 1:
                attr_dict = {'data': h5f[group][key][:][0],
                             'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                ds.attrs[key] = attr_dict
            elif h5f[group][key].ndim == 2:
                attr_dict = {'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                da = xr.DataArray(h5f[group][key][:],
                                  dims={'cross_track': np.arange(h5f[group][key].shape[0]),
                                        'along_track': np.arange(h5f[group][key].shape[1])},
                                  attrs=attr_dict)
                ds[key] = da

            elif h5f[group][key].ndim == 3:
                try:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    time3d = get_time(h5f[group]['scantime'][:], numbers=h5f[group][key].shape[0])
                    da = xr.DataArray(h5f[group][key][:],
                                      dims={'range': np.arange(h5f[group][key].shape[0]),
                                            'cross_track': np.arange(h5f[group][key].shape[1]),
                                            'along_track': np.arange(h5f[group][key].shape[2])},
                                      coords={
                                          'lon3d': (['range', 'cross_track', 'along_track'], h5f[group]['lon3D'][:]),
                                          'lat3d': (['range', 'cross_track', 'along_track'], h5f[group]['lat3D'][:]),
                                          'time3d': (['range', 'cross_track', 'along_track'], time3d),
                                          'alt3d': (['range', 'cross_track', 'along_track'], h5f[group]['alt3D'][:])},
                                      attrs=attr_dict)
                    ds[key] = da
                except ValueError:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(h5f[group][key][:],
                                      dims={'vector': np.arange(h5f[group][key].shape[0]),
                                            'cross_track': np.arange(h5f[group][key].shape[1]),
                                            'along_track': np.arange(h5f[group][key].shape[2])},
                                      attrs=attr_dict)
                    ds[key] = da
        ds_res[group] = ds
    return ds_res


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    ds = hdf2xr(files[-2])
    print(ds)


if __name__ == '__main__':
    main()
    pass
