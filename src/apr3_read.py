#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import h5py
import xarray as xr
import numpy as np
from utils import get_pars_from_ini


def hdf2xr(h5_path, groups=None, campaign='Camp2ex'):
    """
    Function that converts CAMP2EX files (hdf5 files) to xarray datasets
    :param groups: list of Dataset to retrieve e.g. ['lores', 'hires']. If None, all groups will be retrieved.
    :param h5_path: full path to the hdf5 file
    :return: a dictionary with groups a key and datasets as values
    """
    dt_params = get_pars_from_ini()
    h5f = h5py.File(h5_path, mode='r')
    if not groups:
        groups = [i[0] for i in h5f.items()]
    members = {i: [j[0] for j in h5f.get(i).items()] for i in groups}
    ds_res = {}
    for group in groups:
        ds = xr.Dataset()
        for key in members[group]:
            if h5f[group][key].size == 1:
                attr_dict = {'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                ds[key] = xr.DataArray(h5f[group][key][:][0],
                                       attrs=attr_dict)
            elif h5f[group][key].ndim == 2:
                attr_dict = {'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                da = xr.DataArray(h5f[group][key][:],
                                  dims={'cross_track': np.arange(h5f[group][key].shape[0]),
                                        'along_track': np.arange(h5f[group][key].shape[1])},
                                  attrs=attr_dict)
                if key == 'roll':
                    ds['roll_'] = da
                else:
                    ds[key] = da

            elif h5f[group][key].ndim == 3:
                try:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(h5f[group][key][:],
                                      dims={'range': np.arange(h5f[group][key].shape[0]),
                                            'cross_track': np.arange(h5f[group][key].shape[1]),
                                            'along_track': np.arange(h5f[group][key].shape[2])},
                                      coords={
                                          'lon3d': (['range', 'cross_track', 'along_track'], h5f[group]['lon3D'][:]),
                                          'lat3d': (['range', 'cross_track', 'along_track'], h5f[group]['lat3D'][:]),
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
    del h5f
    return ds_res


def main():
    import time
    now = time.time()
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    for i, file in enumerate(files):
        ds = hdf2xr(file, groups=['lores'])
        ds['lores'].to_zarr(store=f'/home/alfonso/Documents/camp2ex_proj/zarr/apr3.zarr', mode='a',
                            append_dim='along_track')
    print(time.time() - now)


if __name__ == '__main__':
    main()
    pass
