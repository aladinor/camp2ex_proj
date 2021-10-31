#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import dask
import pandas as pd
import xarray as xr
import dask.array as da
import numpy as np
from re import split
from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree

from skimage.filters import gaussian, threshold_otsu
from skimage import measure
from dask import delayed
from dask_image.ndfilters import uniform_filter as uf
from dask_image.ndmeasure import variance as varian

# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client, progress

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
path_proj = get_pars_from_ini(campaign='loc')[location]['path_proj']


def get_col_row(x, size=30):
    ncols = x.ptp() / size
    return int(ncols)


def excluding_mesh(x, y, nx=30, ny=30):
    """
    Construct a grid of points, that are some distance away from points (x,
    """

    dx = x.ptp() / nx
    dy = y.ptp() / ny

    xp, yp = np.mgrid[x.min() - 2 * dx:x.max() + 2 * dx:(nx + 2) * 1j,
             y.min() - 2 * dy:y.max() + 2 * dy:(ny + 2) * 1j]
    xp = xp.ravel()
    yp = yp.ravel()

    tree = KDTree(np.c_[x, y])
    dist, j = tree.query(np.c_[xp, yp], k=1)

    # Select points sufficiently far away
    m = (dist > np.hypot(dx, dy))
    return xp[m], yp[m]


def regridd(data, x, y, size=30):
    """
    data = xarray datarray
    size = desired pixel size in meters
    """
    if data.ndim > 2:
        x_n = np.rollaxis(x.reshape(-1, x.shape[-1]), 1)
        y_n = np.rollaxis(y.reshape(-1, y.shape[-1]), 1)
        z_s = data.compute()
        z_n = np.rollaxis(z_s.reshape(-1, z_s.shape[-1]), 1)
        idx_n = x_n.argsort(axis=-1)
        x_n = np.take_along_axis(x_n, idx_n, axis=-1)
        y_n = np.take_along_axis(y_n, idx_n, axis=-1)
        z_n = np.take_along_axis(z_n, idx_n, axis=-1)
        ncols_n = max(np.apply_along_axis(get_col_row, arr=x_n, axis=1))
        nrows_n = max(np.apply_along_axis(get_col_row, arr=y_n, axis=1))
        vp_n = [delayed(excluding_mesh)(x_n[i], y_n[i]) for i in range(x_n.shape[0])]
        vp_n = da.rollaxis(da.dstack(dask.compute(*vp_n)), -1)
        xp_n, yp_n = vp_n[:, 0], vp_n[:, 1]
        zp_n = [delayed(da.zeros_like)(xp_n[i]) for i in range(xp_n.shape[0])]
        zp_n = da.rollaxis(da.dstack(dask.compute(*zp_n))[0], -1)
        x_new_n = da.from_array(np.rollaxis(np.linspace(np.amin(x_n, -1), np.amax(x_n, -1), ncols_n), 1))
        y_new_n = da.from_array(np.rollaxis(np.linspace(np.amax(y_n, -1), np.amin(y_n, -1), nrows_n), 1))
        mesh = [delayed(da.meshgrid)(x_new_n[i], y_new_n[i]) for i in range(x_new_n.shape[0])]
        mesh = dask.compute(*mesh)
        xi = da.asarray(mesh)[:, 0]
        yi = da.asarray(mesh)[:, 1]
        z0 = [delayed(griddata)((np.r_[x_n[i, :], xp_n[i]], np.r_[y_n[i, :], yp_n[i]]), np.r_[z_n[i, :], zp_n[i]],
                                (xi[i], yi[i]), method='linear', fill_value=-9999)
              for i in range(xi.shape[0])]
        z0 = da.dstack(dask.compute(*z0))
        return z0, da.rollaxis(da.rollaxis(xi, axis=-1), axis=-1), da.rollaxis(da.rollaxis(yi, axis=-1), axis=-1)
    else:
        x_s = x.flatten()
        y_s = y.flatten()
        z_s = data.compute().flatten()
        idx = x_s.argsort()
        x_s, y_s = np.take_along_axis(x_s, idx, axis=0), np.take_along_axis(y_s, idx, axis=0)
        z_s = np.take_along_axis(z_s, idx, axis=0)
        ncols = get_col_row(x=x_s, size=size)
        nrows = get_col_row(x=y_s, size=size)
        x_new = np.linspace(x_s.min(), x_s.max(), int(ncols))
        y_new = np.linspace(y_s.max(), y_s.min(), int(nrows))
        xi, yi = np.meshgrid(x_new, y_new)
        xp, yp = excluding_mesh(x_s, y_s, nx=35, ny=35)
        zp = np.nan + np.zeros_like(xp)
        z0 = griddata((np.r_[x_s, xp], np.r_[y_s, yp]), np.r_[z_s, zp], (xi, yi), method='linear', fill_value=-9999)
        return z0, xi, yi


def lee_filter_new(img, size, tresh=-150):
    if img.ndim == 2:
        shape = (size, size)
    else:
        shape = (size, size, 1)
    img = da.where(da.logical_or(da.isnan(img), da.equal(img, -9999)), tresh, img)
    img_mean = uf(img, shape)
    img_sqr_mean = uf(da.power(img, 2), shape)
    img_variance = img_sqr_mean - da.power(img_mean, 2)
    overall_variance = varian(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    img_output = da.where(img_output > 0, img_output, 0)
    return img_output


def process_new(zhh14, x, y, time):
    x = x[:, 0, :, :]
    img_filtered = lee_filter_new(zhh14, size=3, tresh=-200)
    img, xi, yi = regridd(img_filtered, x, y)
    if zhh14.ndim > 2:
        total, _x, _y = regridd(img_filtered[:, :, 0], x[:, :, 0], y[:, :, 0])
        total = np.nansum(np.where(total >= 0, 1, 0), axis=1)
        total = np.repeat(total[:, np.newaxis], img_filtered.shape[-1], 1)
    else:
        total = np.nansum(np.where(img >= 0, 1, 0), axis=1)  # Total of number of pixels
    num_pixels = np.nansum(np.where(img > 0, 1, np.nan), axis=1)
    weigt = num_pixels / total
    img = np.where(img > 0., img, 0.)
    blurred = gaussian(img, sigma=0.8)
    binary = blurred > threshold_otsu(blurred)
    labels = measure.label(binary)
    if labels.ndim > 2:
        props = [measure.regionprops(labels[:, :, i]) for i in range(labels.shape[-1])]
        _props_all = [[[j.area for j in prop], [j.perimeter for j in prop], [j.major_axis_length for j in prop],
                       [j.minor_axis_length for j in prop], [j.bbox for j in prop]] for prop in props]
        df = pd.DataFrame(data=_props_all, columns=['area', 'perimeter', 'axmax', 'axmin', 'bbox'],
                          index=pd.to_datetime(time))
    else:
        props = measure.regionprops(labels)
        _props_all = [[[prop.area], [prop.perimeter], [prop.major_axis_length], [prop.minor_axis_length],
                       [prop.bbox]] for prop in props]
        df = pd.DataFrame(data=_props_all, columns=['area', 'perimeter', 'axmax', 'axmin', 'bbox'], index=time)
    df['total'] = [[total[:, i]] for i in range(total.shape[-1])]
    df['num_px'] = [[num_pixels.compute()[:, i]] for i in range(num_pixels.shape[-1])]
    df['weigt'] = [[weigt.compute()[:, i]] for i in range(weigt.shape[-1])]
    # df = df.explode(['area', 'perimeter', 'axmax', 'axmin'])
    # df.to_csv(f'../results/all_{len(time)}.csv')
    # df = df.astype(dtype={'area': 'float', 'perimeter': 'float', 'axmax': 'float', 'axmin': 'float'})
    # df = df[df.area > 50.0]
    # df_new = pd.DataFrame(index=time, data=np.full(len(time), np.nan), columns=['area'])
    # df_new = df_new.merge(df, left_index=True, right_index=True, how='left').drop(['area_x'], axis=1)
    # idx = df_new.index.duplicated()
    xr_prop = xr.Dataset.from_dataframe(df).rename_dims({'index': 'time'}).rename({'index': 'time'})
    # xr_prop['x'] = xr.DataArray(xi, dims=['cross_track', 'height', 'time'],
    #                             coords={'x': (['cross_track', 'height', 'time'], x)})
    # xr_prop['y'] = xr.DataArray(yi, dims={'time': time})
    return xr_prop.area, xr_prop.perimeter, xr_prop.axmax, xr_prop.axmin, xr_prop.bbox


def ufunc_wrapper(data):
    x = data.range * data.DR * np.sin(np.deg2rad(data.azimuth))  # add roll
    y = data.alt3d * np.cos(np.deg2rad(data.azimuth))
    zhh = data.zhh14.where(data.alt3d > 500)
    _data = [zhh, x, y, data.time]
    icd = [list(i.dims) for i in _data]
    dfk = {'allow_rechunk': True, 'output_sizes': {}}
    a, p, mx, mn, bbox = xr.apply_ufunc(process_new,
                                        *_data,
                                        input_core_dims=icd,
                                        output_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"]],
                                        dask_gufunc_kwargs=dfk,
                                        dask='parallelized',
                                        vectorize=True,
                                        output_dtypes=[(object), (object), (object), (object), (object)]
                                        )
    ds_out = a.to_dataset(name='area')
    ds_out['perimeter'] = p
    ds_out['ax_max'] = mx
    ds_out['ax_min'] = mn
    ds_out['bbox'] = bbox
    return ds_out


def main():
    ds_xr = xr.open_zarr(f'{path_data}/zarr/KUsKAs_Wn/lores.zarr', consolidated=True)
    ds_xr = ds_xr.sel(time=~ds_xr.get_index("time").duplicated())
    # ds_data = ds_xr[['zhh14', 'azimuth', 'DR']].sel(time='2019-09-16 03:12:58')
    ds_data = ds_xr[['zhh14', 'azimuth', 'DR']].sel(time=slice('2019-09-16 03:12:50', '2019-09-16 03:13:05'))
    a = ufunc_wrapper(ds_data)
    w = dask.compute(a)
    df = w[0].to_dataframe()
    template = ds_xr.time.isel(time=0)
    # mapped = xr.map_blocks(process_map, ds_zhh, template=template).compute()
    # a = process_new(ds_zhh)
    # print(a.values)
    pass


if __name__ == '__main__':
    main()
