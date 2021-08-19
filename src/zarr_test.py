#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import time
import xarray as xr
from apr3_read import hdf2xr
from utils import make_dir


def data_test(path_file):
    ds_xr = xr.open_zarr(f'{path_file}/zarr/apr3.zarr', consolidated=True, decode_times=True)
    print('Done!')


def hdf2zar(path_file):
    now = time.time()
    files_dict = {'Wn': glob.glob(f'{path_file}/data/*_Wn.h5'),
                  'KUsKAsWs': glob.glob(f'{path_file}/data/*_KUsKAsWs.h5'),
                  'KUsKAs_Wn': glob.glob(f'{path_file}/data/*_KUsKAs.h5') + glob.glob(
                      f'{path_file}/data/*_KUsKAsWn.h5')}

    for key_outer in files_dict:
        files = files_dict[key_outer]
        files.sort()
        make_dir(f'{path_file}/zarr/{key_outer}')
        with open(f'{path_file}/zarr/{key_outer}/good_files.txt', 'w') as good,  \
                open(f'{path_file}/zarr/{key_outer}/bad_files.txt', 'w') as bad:
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
                        ds[key].to_zarr(store=f'{path_file}/zarr/{key_outer}/{key}.zarr', **args)
                        good.writelines(f"{file.split('/')[-1]}, {ds.keys()}, {key}, \n")
                    except (ValueError, KeyError, IndexError) as e:
                        bad.writelines(f"{file.split('/')[-1]}, {ds.keys()}, {key}, {e}\n")
                del ds
        good.close()
        bad.close()
    print(time.time() - now)


def main():
    path_file = '/media/alfonso/drive/Alfonso/camp2ex_proj'
    hdf2zar(path_file)
    # data_test(path_file)


if __name__ == '__main__':
    main()


