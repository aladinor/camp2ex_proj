#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import dask.array as da
import numpy as np
from re import split
from scipy.spatial import cKDTree as KDTree

from skimage.filters import gaussian, threshold_otsu
from skimage import measure
from scipy.interpolate import griddata
from dask_image.ndfilters import uniform_filter as uf
from dask_image.ndmeasure import variance as varian
import matplotlib.patches as mpatches

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

    # Use KDTree to answer the question: "which point of set (x,y) is the
    # nearest neighbors of those in (xp, yp)"
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
        x_s = np.asarray([x[:, :, i].flatten() for i in range(x.shape[-1])])
        y_s = np.asarray([y[:, :, i].flatten() for i in range(y.shape[-1])])
        z_s = data.compute()
        z_s = np.asarray([z_s[:, :, i].flatten() for i in range(z_s.shape[-1])])
        idx = np.asarray([x_s[i].argsort() for i in range(x_s.shape[0])])
        x_s = np.asarray([np.take_along_axis(x_s[i], idx[i], axis=0) for i in range(x_s.shape[0])])
        y_s = np.asarray([np.take_along_axis(y_s[i], idx[i], axis=0) for i in range(y_s.shape[0])])
        z_s = np.asarray([np.take_along_axis(z_s[i], idx[i], axis=0) for i in range(z_s.shape[0])])
        ncols = max(np.apply_along_axis(get_col_row, arr=x_s, axis=1))
        nrows = max(np.apply_along_axis(get_col_row, arr=y_s, axis=1))
        vp = np.asarray([excluding_mesh(x_s[i], y_s[i]) for i in range(x_s.shape[0])], dtype=object)
        xp, yp = vp[:, 0], vp[:, 1]
        zp = np.asarray([np.nan + np.zeros_like(xp[i]) for i in range(xp.shape[0])], dtype=object)
        x_new = np.asarray([np.linspace(x_s[i, :].min(), x_s[i, :].max(), ncols) for i in range(xp.shape[0])])
        y_new = np.asarray([np.linspace(y_s[i, :].max(), y_s[i, :].min(), nrows)
                            for i in range(xp.shape[0])])
        mesh = [np.meshgrid(x_new[i], y_new[i]) for i in range(x_new.shape[0])]
        xi = np.asarray(mesh)[:, 0]
        yi = np.asarray(mesh)[:, 1]
        z0 = [griddata((np.r_[x_s[i, :], xp[i]], np.r_[y_s[i, :], yp[i]]), np.r_[z_s[i, :], zp[i]],
                       (xi[i], yi[i]), method='linear', fill_value=-9999)
              for i in range(xi.shape[0])]
        z0 = np.dstack(z0)
        return z0
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
        return z0


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


def process_new(zhh14, _range, azimuth, alt3d, bin_size, time):
    x = _range * bin_size.values[0] * np.sin(np.deg2rad(azimuth))  # add roll
    y = alt3d * np.cos(np.deg2rad(azimuth.T))
    img_filtered = lee_filter_new(zhh14, size=3, tresh=-200)
    img = regridd(img_filtered, x.values, y.values)
    # img = lee_filter_new(img, size=5, tresh=-300)
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
        df = pd.DataFrame(data=_props_all, columns=['area', 'perimeter', 'axmax', 'axmin', 'bbox'])


    if img.ndim > 2:
        shp = img.shape[-1]
        for i in range(shp):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            # img_plot = np.where(img[i] > 0, img[i], np.nan)
            img_plot = img[:, :, i]
            ax1.pcolormesh(x.isel(time=i), y.isel(time=i), img_filtered.compute()[:, :, i], cmap='jet',
                           vmax=40, vmin=0, shading='auto')
            ax2.imshow(img_plot, aspect='auto', cmap='jet', vmax=40, vmin=0)
            for region in props[i]:
                # take regions with large enough areas
                if region.area >= 100:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax2.add_patch(rect)
            plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        # img_plot = np.where(img[i] > 0, img[i], np.nan)
        img_plot = img
        ax1.pcolormesh(x, y, img_filtered.compute(), cmap='jet', vmax=40, vmin=0)
        ax2.imshow(img_plot, aspect='auto', cmap='jet', vmax=40, vmin=0)
        for region in props:
            # take regions with large enough areas
            if region.area >= 100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax2.add_patch(rect)

        ax2.set_axis_off()
        plt.tight_layout()
        plt.show()


def main():
    ds_xr = xr.open_zarr('C:/Users/alfonso8/Documents/python/camp2ex_proj/zarr/lores.zarr', consolidated=True)
    ds_xr = ds_xr.sel(time=~ds_xr.get_index("time").duplicated())
    # ds_data = ds_xr[['zhh14', 'azimuth', 'DR']].sel(time='2019-09-16 03:12:58').isel(time=0)
    ds_data = ds_xr[['zhh14', 'azimuth', 'DR']].sel(time=slice('2019-09-16 03:12:50', '2019-09-16 03:13:05'))
    # ds_data = ds_xr[['zhh14', 'azimuth', 'DR']].isel(time=slice(178, 190))
    ds_zhh = ds_data.zhh14.where(ds_data.alt3d > 500)
    process_new(zhh14=ds_zhh, _range=ds_data.range, azimuth=ds_data.azimuth,
                alt3d=ds_data.alt3d, bin_size=ds_data.DR, time=ds_data.time)
    pass


if __name__ == '__main__':
    main()
