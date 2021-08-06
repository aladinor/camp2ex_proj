#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import time
from apr3_read import hdf2xr
import xarray as xr
from sand_box import plot_multi_panel
from utils import get_time


def data_test(path_file):
    ds_xr = xr.open_zarr(f'{path_file}/zarr/apr3.zarr', consolidated=True)

    time3d = get_time(time_array=ds_xr.scantime.dropna(dim='along_track', how='any').values,
                      numbers=ds_xr.zhh14.shape[0])
    plot_multi_panel(lon=ds_xr.lon.dropna(dim='along_track', how='all'),
                     lat=ds_xr.lat.dropna(dim='along_track', how='all'),
                     dbz=ds_xr.zhh14.dropna(dim='along_track', how='all'),
                     alt3d=ds_xr.alt3d.dropna(dim='along_track', how='all'),
                     rang=ds_xr.range,
                     azimt=ds_xr.azimuth.dropna(dim='along_track', how='all'),
                     time3d=time3d,
                     s0hh14=ds_xr.s0hh14.dropna(dim='along_track', how='all'))
    print('Done!')


def hdf2zar(path_file):
    now = time.time()
    files = glob.glob(f'{path_file}/data/*KUsKAsWn.h5')
    with open(f'{path_file}/good_files.txt', 'w') as good,  \
            open(f'{path_file}/bad_files.txt', 'w') as bad:
        for i, file in enumerate(files):
            ds = hdf2xr(file, groups=['lores'])
            args = {'consolidated': True}
            if i == 0:
                args['mode'] = 'w'
            else:
                args['mode'] = 'a'
                args['append_dim'] = 'along_track'
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


