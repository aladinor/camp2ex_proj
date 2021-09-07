#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import h5py
import xarray as xr
import numpy as np
from src.utils import get_pars_from_ini, get_time, make_dir
from zarr import Blosc


def hdf2xr(h5_path, groups=None, campaign='camp2ex'):
    """
    Function that converts CAMP2EX files (hdf5 files) to xarray datasets
    :param campaign: campaign from where data comes from
    :param groups: list of Dataset to retrieve e.g. ['lores', 'hires']. If None, all groups will be retrieved.
    :param h5_path: full path to the hdf5 file
    :return: a dictionary with groups as keys and datasets as values
    """
    dt_params = get_pars_from_ini(campaign=campaign)
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


def hdf2zar(path_files):
    """
    Functions that converts APR3 hdf5 files into zarr files
    :param path_files: full path to APR3 hdf5 files. Files must be inside a data folder. e.g. '/{path_files}/data'
    :return: Create zarr files and a txt files with "good" and "bad" files
    """
    files_dict = {'Wn': glob.glob(f'{path_files}/data/*_Wn.h5'),
                  'KUsKAsWs': glob.glob(f'{path_files}/data/*_KUsKAsWs.h5'),
                  'KUsKAs_Wn': glob.glob(f'{path_files}/data/*_KUsKAs.h5') + glob.glob(
                      f'{path_files}/data/*_KUsKAsWn.h5')}

    for key_outer in files_dict:
        files = files_dict[key_outer]
        files.sort()
        if not files:
            continue
        make_dir(f'{path_files}/zarr/{key_outer}')
        with open(f'{path_files}/zarr/{key_outer}/good_files.txt', 'w') as good,  \
                open(f'{path_files}/zarr/{key_outer}/bad_files.txt', 'w') as bad:
            for i, file in enumerate(files):
                print(i, file)
                ds = hdf2xr(file)
                args = {'consolidated': True}
                for key in ds.keys():
                    if i == 0:
                        args['mode'] = 'w'
                    else:
                        args['mode'] = 'a'
                        if not hasattr(ds[key], 'time'):
                            args['append_dim'] = 'params'
                        else:
                            args['append_dim'] = 'time'
                    try:
                        ds[key].to_zarr(store=f'{path_files}/zarr/{key_outer}/{key}.zarr', **args)
                        good.writelines(f"{file.split('/')[-1]}, {ds.keys()}, {key}, \n")
                    except ValueError as e:
                        tries = 0
                        while e is not True and not tries > 3:
                            _var, _num = str(e).split(' ')[1][1:-1], str(e).split(' ')[14][:-1]
                            args = {'consolidated': True}
                            if os.path.isdir(f'{path_files}/zarr/{key_outer}/{key}_{_var}_{_num}.zarr'):
                                args['mode'] = 'a'
                                if not hasattr(ds[key], 'time'):
                                    args['append_dim'] = 'params'
                                else:
                                    args['append_dim'] = 'time'
                            else:
                                args['mode'] = 'w'
                            ds[key].to_zarr(store=f'{path_files}/zarr/{key_outer}/{key}_{_var}_{_num}.zarr', **args)
                            tries += 1
                            e = True
                            good.writelines(f"{file.split('/')[-1]}, {ds.keys()}, {key}, \n")
                        bad.writelines(f"{file.split('/')[-1]}, {ds.keys()}, {key}, {e}\n")
                del ds
        good.close()
        bad.close()


def main():
    pass


if __name__ == '__main__':
    main()
    pass
