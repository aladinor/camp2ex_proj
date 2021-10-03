#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import sys
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from re import split
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini


def rechunk(path, path_zarr):
    ds_xr = xr.open_zarr(path_zarr, consolidated=True)
    path_save = f"{path}/zarr_rechunked/{path_zarr.split('/')[-2]}/{path_zarr.split('/')[-1]}"
    ds_xr.to_zarr(f'{path_save}', chunk_store={'range': -1, 'cross_track': -1, 'time': 10},
                  consolidated=True, mode='w', compute=True)
    del ds_xr


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    folders = [i for i in glob.glob(f'{path_data}/zarr/*/*') if i.endswith('.zarr')]
    cluster = LocalCluster()
    client = Client(cluster)
    res = []
    for i in folders:
        y = dask.delayed(rechunk)(path_data, i)
        res.append(y)
    results = dask.compute(*res)
    print('termine')
    pass


if __name__ == '__main__':
    main()
