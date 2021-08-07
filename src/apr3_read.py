#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
        diff = [i for i in members[group] + list(dt_params[group].keys())
                if i not in members[group] or i not in list(dt_params[group].keys())]
        members[group] = members[group] + diff
        ds = xr.Dataset()
        for key in members[group]:
            try:
                if h5f[group][key].size == 1:
                    attr_dict = {'data': h5f[group][key][:][0][0],
                                 'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    ds.attrs[key] = attr_dict

                elif h5f[group][key].ndim == 2:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(h5f[group][key][:],
                                      coords={'cross_track': np.arange(h5f[group][key].shape[0]),
                                              'along_track': np.arange(h5f[group][key].shape[1])},
                                      dims=['cross_track', 'along_track'],
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
                        flag = key
                    except ValueError:
                        attr_dict = {'units': dt_params[group][key]['units'],
                                     'notes': dt_params[group][key]['notes']}
                        da = xr.DataArray(h5f[group][key][:],
                                          dims={'vector': np.arange(h5f[group][key].shape[0]),
                                                'cross_track': np.arange(h5f[group][key].shape[1]),
                                                'along_track': np.arange(h5f[group][key].shape[2])},
                                          attrs=attr_dict)
                        ds[key] = da
            except KeyError:
                attr_dict = {'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                da = xr.DataArray(np.full_like(h5f[group][flag][:], np.nan),
                                  dims={'range': np.arange(h5f[group][flag].shape[0]),
                                        'cross_track': np.arange(h5f[group][flag].shape[1]),
                                        'along_track': np.arange(h5f[group][flag].shape[2])},
                                  coords={
                                      'lon3d': (['range', 'cross_track', 'along_track'], h5f[group]['lon3D'][:]),
                                      'lat3d': (['range', 'cross_track', 'along_track'], h5f[group]['lat3D'][:]),
                                      'alt3d': (['range', 'cross_track', 'along_track'], h5f[group]['alt3D'][:])},
                                  attrs=attr_dict)
                ds[key] = da
        ds_res[group] = ds
    del h5f
    return ds_res


def main():
    pass


if __name__ == '__main__':
    main()
    pass
