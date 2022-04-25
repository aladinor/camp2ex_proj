#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import xarray as xr
import numpy as np
import zarr
from src.utils import get_pars_from_ini, get_time
from zarr import Blosc


def hdf2xr(h5_path, groups=None, campaign='camp2ex'):
    """
    Function that converts CAMP2EX files (hdf5 files) to xarray datasets
    :param campaign: campaign from where data comes from
    :param groups: list of Dataset to retrieve e.g. ['lores', 'hires']. If None, all groups will be retrieved.
    :param h5_path: full path to the hdf5 file
    :return: a dictionary with groups as keys and datasets as values
    """
    dt_params = get_pars_from_ini(file_name=campaign)
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
            compressor = Blosc(cname="lz4", clevel=5, shuffle=0)
            encode = {'_FillValue': '-9999', 'compresor': compressor}
            try:
                time = get_time(h5f[group]['scantime'][12, :])
            except (IndexError, ValueError):
                time = get_time(h5f[group]['scantime'][0, :])
            except KeyError:
                try:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(h5f[group][key][:][0],
                                      dims=['params'],
                                      attrs=attr_dict)
                    encode['dtype'] = h5f[group][key][:][0].dtype
                    da.encoding = encode
                    ds[key] = da
                    flag_1 = key
                except KeyError:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(np.full_like(h5f[group][flag_1][:][0], np.nan),
                                      dims=['params'],
                                      attrs=attr_dict)
                    encode['dtype'] = h5f[group][flag_1][:][0].dtype
                    da.encoding = encode
                    ds[key] = da
                continue

            try:
                if h5f[group][key].size == 1:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}
                    da = xr.DataArray(h5f[group][key][:][0],
                                      dims=['bin_size'],
                                      attrs=attr_dict)
                    encode['dtype'] = h5f[group][key][:][0].dtype
                    da.encoding = encode
                    ds[key] = da
                elif h5f[group][key].ndim == 2:
                    attr_dict = {'units': dt_params[group][key]['units'],
                                 'notes': dt_params[group][key]['notes']}

                    da = xr.DataArray(h5f[group][key][:],
                                      coords={'cross_track': np.arange(h5f[group][key].shape[0]),
                                              'time': time},
                                      dims=['cross_track', 'time'],
                                      attrs=attr_dict)
                    encode['dtype'] = h5f[group][key][:][0].dtype
                    da.encoding = encode
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
                                                'time': time},
                                          coords={
                                              'lon3d': (['range', 'cross_track', 'time'], h5f[group]['lon3D'][:]),
                                              'lat3d': (['range', 'cross_track', 'time'], h5f[group]['lat3D'][:]),
                                              'alt3d': (['range', 'cross_track', 'time'], h5f[group]['alt3D'][:])},
                                          attrs=attr_dict)
                        encode['dtype'] = h5f[group][key][:][0].dtype
                        da.encoding = encode
                        ds[key] = da
                        flag = key
                    except ValueError:
                        attr_dict = {'units': dt_params[group][key]['units'],
                                     'notes': dt_params[group][key]['notes']}
                        da = xr.DataArray(h5f[group][key][:],
                                          dims={'vector': np.arange(h5f[group][key].shape[0]),
                                                'cross_track': np.arange(h5f[group][key].shape[1]),
                                                'time': time},
                                          attrs=attr_dict)
                        encode['dtype'] = h5f[group][key][:][0].dtype
                        da.encoding = encode
                        ds[key] = da
            except KeyError:
                attr_dict = {'units': dt_params[group][key]['units'],
                             'notes': dt_params[group][key]['notes']}
                da = xr.DataArray(np.full_like(h5f[group][flag][:], np.nan),
                                  dims={'range': np.arange(h5f[group][flag].shape[0]),
                                        'cross_track': np.arange(h5f[group][flag].shape[1]),
                                        'time': time},
                                  coords={
                                      'lon3d': (['range', 'cross_track', 'time'], h5f[group]['lon3D'][:]),
                                      'lat3d': (['range', 'cross_track', 'time'], h5f[group]['lat3D'][:]),
                                      'alt3d': (['range', 'cross_track', 'time'], h5f[group]['alt3D'][:])},
                                  attrs=attr_dict)
                encode['dtype'] = h5f[group][flag][:][0].dtype
                da.encoding = encode
                ds[key] = da
            if hasattr(ds, 'time'):
                encode['dtype'] = 'int64'
                ds.time.encoding = encode
        ds_res[group] = ds
    del h5f
    return ds_res


def ds2zar(ds, store='../zarr/camp2ex.zarr'):
    """
    Functions that converts APR3 hdf5 files into zarr files
    :param store: zarr store path
    :param ds: xarray dataset to be stored
    :return : Creates zarr files in a given location
    """
    store = zarr.DirectoryStore(store)
    for key in ds.keys():
        print(key)
        args = {'consolidated': True}
        try:
            ds[key].to_zarr(store=store, group=key, **args)
        except zarr.errors.ContainsGroupError:
            args['mode'] = 'a'
            if not hasattr(ds[key], 'time'):
                args['append_dim'] = 'params'
            else:
                args['append_dim'] = 'time'

            ds[key].to_zarr(store=store, group=key, **args)


def main():
    pass


if __name__ == '__main__':
    main()
    pass
