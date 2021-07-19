import glob
import drpy
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from apr3_read import hdf2xr
from matplotlib.gridspec import GridSpec


def plot_multi_panel(lores):
    for i in range(lores.zhh35.shape[2]):
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[-1, :])

        lon = lores.lon[12, :].values.copy()
        lat = lores.lat[12, :].values.copy()
        ax1.set_extent([lon.max() + 2, lon.min() - 2, lat.min() - 2, lat.max() + 2], crs=ccrs.Geodetic())
        lon[i + 1:] = np.nan
        lat[i + 1:] = np.nan
        track = sgeom.LineString(zip(lon, lat))
        ax1.add_geometries([track], ccrs.PlateCarree(), facecolor='none', edgecolor='blue',
                           linewidth=2)
        ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray',
                      alpha=0.5, linestyle='--')
        ax1.coastlines()
        dbz = lores.zhh35.values.copy()
        x = lores.range[:] * np.sin(np.deg2rad(lores.azimuth[:]))
        # y = lores.range[:] * np.cos(np.deg2rad(lores.azimuth[:]))  # for plane relative coordinates
        y = lores.alt3D.values * np.cos(np.deg2rad(lores.azimuth.values))
        a = ax2.pcolormesh(x[:, :, i], y[:, :, i], dbz[:, :, i],
                           cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=-10, vmax=40)
        # ax2.invert_yaxis()  # for plane relative coordinates
        plt.colorbar(a, ax=ax2, orientation='vertical')
        dbz[:, 12, i:] = np.nan
        c = ax3.pcolormesh(lores.time3d[:, 12, :], lores.alt3D[:, 12, :], dbz[:, 12, :],
                           cmap=drpy.graph.cmaps.HomeyerRainbow, vmin=0, vmax=40)
        plt.colorbar(c, ax=ax3)
        # plt.show()
        plt.savefig(f'../results/Ka/ref/dual_plot_2_{i:03}')


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    for file in files[-2:-1]:
        ds_dict = hdf2xr(file, groups=['lores'])
        lores = ds_dict['lores']
        plot_multi_panel(lores)
        print('pause')


if __name__ == '__main__':
    main()
