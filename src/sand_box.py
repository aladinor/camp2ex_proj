import glob
import drpy
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from utils import time_3d
from matplotlib.gridspec import GridSpec
from datetime import datetime


def plot_multi_panel(ds, time):
    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[-1, :])

    ax1.set_extent([ds.lon.max() + 2, ds.lon.min() - 2, ds.lat.min() - 2, ds.lat.max() + 2], crs=ccrs.Geodetic())
    if time == ds.time.min():
        _lon = ds.lon.sel(time=slice(ds.time.min(), time))
        _lat = ds.lat.sel(time=slice(ds.time.min(), time))
    else:
        _lon = ds.lon.sel(time=slice(ds.time.min(), time)).values[12, :]
        _lat = ds.lat.sel(time=slice(ds.time.min(), time)).values[12, :]
    track = sgeom.LineString(zip(_lon, _lat))
    # track = sgeom.LineString(zip(ds.lon[12, :], ds.lat[12, :]))
    ax1.add_geometries([track], ccrs.PlateCarree(), facecolor='none', edgecolor='blue',
                       linewidth=2)
    ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray',
                  alpha=0.5, linestyle='--')
    ax1.coastlines()
    x = ds.range * np.sin(np.deg2rad(ds.azimuth.sel(time=time)))
    y = ds.alt3D.sel(time=time) * np.cos(np.deg2rad(ds.azimuth.sel(time=time)))
    a = ax2.pcolormesh(x, y, ds.zhh14.sel(time=time), cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=-10, vmax=40)
    # ax2.invert_yaxis()  # for plane relative coordinates
    plt.colorbar(a, ax=ax2, orientation='vertical')
    ax2.set_ylim(0, y.max())
    times_3d = time_3d(ds.scantime.sel(time=slice(ds.time.min(), time)), ds.alt3D.shape[0])
    c = ax3.pcolormesh(times_3d[:, 12, :],
                       ds.alt3D.sel(time=slice(ds.time.min(), time))[:, 12, :],
                       ds.zhh14.sel(time=slice(ds.time.min(), time))[:, 12, :],
                       cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=0, vmax=40)
    plt.colorbar(c, ax=ax3)
    ax3.set_ylim(0, y.max())
    ax3.set_xlim(ds.time.min(), ds.time.max())
    ax4 = ax3.twinx()
    ax4.plot(times_3d[0, 12, :], ds.s0hh14.sel(time=slice(ds.time.min(), time))[12])
    ax4.set_ylim(ds.s0hh14.min(), ds.s0hh14.max())
    plt.show()


def main():
    now = datetime.now()
    path_file = '/media/alfonso/drive/Alfonso/camp2ex_proj'
    ds_xr = xr.open_zarr(f'{path_file}/zarr/apr3.zarr', consolidated=True, decode_times=True)
    _, index = np.unique(ds_xr['time'], return_index=True)
    ds_xr = ds_xr.sel(time=~ds_xr.get_index("time").duplicated())
    ds_dates = ds_xr.sel(time=slice('2019-09-15 22:12', '2019-09-15 22:14'))
    for _time in ds_dates.time:
        plot_multi_panel(ds_dates, time=_time)
    print(f'execution time: {(now - datetime.now()) / 60}')
    print(1)


if __name__ == '__main__':
    main()
