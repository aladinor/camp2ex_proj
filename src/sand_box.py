import glob
import drpy
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from apr3_read import hdf2xr
from matplotlib.gridspec import GridSpec


def plot_multi_panel(lon, lat, dbz, alt3d, rang, azimt, time3d):
    for i in range(dbz.shape[2]):
        plt.close('all')
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[-1, :])

        lon_i = lon[12, :].values.copy()
        lat_i = lat[12, :].values.copy()
        ax1.set_extent([lon.max() + 2, lon.min() - 2, lat.min() - 2, lat.max() + 2], crs=ccrs.Geodetic())
        lon_i[i + 1:] = np.nan
        lat_i[i + 1:] = np.nan
        track = sgeom.LineString(zip(lon_i, lat_i))
        ax1.add_geometries([track], ccrs.PlateCarree(), facecolor='none', edgecolor='blue',
                           linewidth=2)
        ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray',
                      alpha=0.5, linestyle='--')
        ax1.coastlines()
        x = rang[0, :] * np.sin(np.deg2rad(azimt))
        # y = lores.range[:] * np.cos(np.deg2rad(lores.azimuth[:]))  # for plane relative coordinates
        y = alt3d * np.cos(np.deg2rad(azimt))
        dbz_i = dbz.values.copy()
        a = ax2.pcolormesh(x[:, :, i], y[:, :, i], dbz_i[:, :, i],
                           cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=-10, vmax=40)
        # ax2.invert_yaxis()  # for plane relative coordinates
        plt.colorbar(a, ax=ax2, orientation='vertical')
        dbz_i[:, 12, i:] = np.nan
        c = ax3.pcolormesh(time3d[:, 12, :], alt3d[:, 12, :], dbz_i[:, 12, :],
                           cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=0, vmax=40)
        plt.colorbar(c, ax=ax3)
        plt.savefig(f'../results/Ku/ref/dual_plot_2_{i:03}')
        # plt.show()


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    lon, lat, dbz, alt3d, rang, azimt, time3d = ([] for _ in range(7))
    for file in files:
        ds_dict = hdf2xr(file, groups=['lores'])
        lores = ds_dict['lores']
        lon.append(lores.lon)
        lat.append(lores.lat)
        dbz.append(lores.zhh14)
        alt3d.append(lores.alt3d)
        rang.append(lores.range)
        azimt.append(lores.azimuth)
        time3d.append(lores.time3d)
        # plot_multi_panel(lores)
    lon = xr.concat(lon, dim='along_track')
    lat = xr.concat(lat, dim='along_track')
    dbz = xr.concat(dbz, dim='along_track')
    alt3d = xr.concat(alt3d, dim='along_track')
    rang = xr.concat(rang, dim='along_track')
    azimt = xr.concat(azimt, dim='along_track')
    time3d = xr.concat(time3d, dim='along_track')
    plot_multi_panel(lon=lon, lat=lat, dbz=dbz, alt3d=alt3d, rang=rang, azimt=azimt, time3d=time3d)
    print('Done. good job!')


if __name__ == '__main__':
    main()
