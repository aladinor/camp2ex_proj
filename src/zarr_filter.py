#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def rolling_window(a, window):
    """ Create a rolling window object for application of functions
    eg: result=np.ma.std(array, 11), 1). """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def texture(fld):
    """ Determine a texture field using an 11pt stdev
    texarray=texture(pyradarobj, field). """
    tex = np.ma.zeros(fld.shape)
    for timestep in range(tex.shape[0]):
        ray = np.ma.std(rolling_window(fld[timestep, :], 3), 1)
        tex[timestep, 1:-1] = ray
        tex[timestep, 0:1] = np.ones(1) * ray[0]
        tex[timestep, -2:] = np.ones(2) * ray[-1]
    return tex


def data_test(path_file):
    ds_xr = xr.open_zarr(f'{path_file}/zarr/KUsKAs_Wn/lores.zarr')

    _events = []
    with open(f'{path_file}/zarr/events.txt', 'w') as events:
        for date in ds_xr.time:
            try:
                masked_data = ds_xr.sel(time=date).where((ds_xr.zhh14.sel(time=date) < 60) &
                                                         (ds_xr.vel14.sel(time=date) < 0))
                tex = texture(masked_data.zhh14.values)
                indicators = (tex > 0).astype(int)
                _sum = np.sum(indicators)
                if _sum > 40:
                    _events.append(date.values)
                    events.writelines(f"{date.values}\n")
            except ValueError:
                continue
    events.close()


def main():
    path_file = '/media/alfonso/drive/Alfonso/camp2ex_proj'
    # path_file = '/data/keeling/a/alfonso8/projects/camp2ex_proj'
    data_test(path_file)


if __name__ == '__main__':
    main()
