#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import time
from apr3_read import hdf2xr
import xarray as xr


def data_test(path_file):
    ds_xr = xr.open_zarr(f'{path_file}/zarr/apr3.zarr', consolidated=True, decode_times=True)
    print('Done!')


def hdf2zar(path_file):
    now = time.time()
    files = glob.glob(f'{path_file}/data/*KUsKAsWn.h5')
    files.sort()
    with open(f'{path_file}/good_files.txt', 'w') as good,  \
            open(f'{path_file}/bad_files.txt', 'w') as bad:
        for i, file in enumerate(files):
            ds = hdf2xr(file, groups=['lores'])
            args = {'consolidated': True}
            if i == 0:
                args['mode'] = 'w'
            else:
                args['mode'] = 'a'
                args['append_dim'] = 'time'
            try:
                ds['lores'].to_zarr(store=f'{path_file}/zarr/apr3.zarr', **args)
                good.write(f"{file.split('/')[-1]}\n")
            except ValueError as e:
                bad.write(f"{file.split('/')[-1]}, {e}\n")
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


