import h5py
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import glob
import numpy as np
import datetime
import xarray as xr


def apr3read(filename):
    """
    Function used for reading apr3 hdf file from CAMP2EX file using Randy's code
    (https://github.com/dopplerchase/Chase_et_al_2018/blob/master/apr3tocit_tools.py)
    :param filename: path and filename of the APR3 HDF file
    :return: dictionary with apr3 data
    """
    hdf = h5py.File(filename, "r")
    desired_data = ['alt3D', 'lat', 'lon', 'scantime', 'surface_index', 'isurf', 'alt_nav', 'zhh14', 'zhh35', 'zhh95',
                    'ldrhh14', 'vel14', 'lon3D', 'lat3D', 'alt3D', 'z95n', 'roll', 'pitch', 'drift', 'z95s', 'z95n']
    apr = {i: hdf['lores'][i][:] for i in hdf['lores'].keys() if i in desired_data}

    flag = 0

    # Not completely sure about this if statement
    if 'z95s' in apr.keys():
        if 'z95n' in apr.keys():
            radar3 = hdf['lores']['z95s']
        else:
            radar3 = hdf['lores']['z95s']
            print('No vv, using hh')
    else:
        radar3 = np.ma.array([])
        flag = 1
        print('No W band')

    # convert time to datetimes
    time_dates = np.asarray([[datetime.datetime.utcfromtimestamp(hdf['lores']['scantime'][:][i, j])
                             for j in range(hdf['lores']['scantime'][:].shape[1])]
                             for i in range(hdf['lores']['scantime'][:].shape[0])])
    # Create a time at each gate (assuming it is the same down each ray, there is a better way to do this)
    time_gate = np.asarray([[[time_dates[i, j] for j in range(time_dates.shape[1])]
                            for i in range(time_dates.shape[0])] for k in range(hdf['lores']['lat3D'][:].shape[0])])

    # Quality control (masked where invalid)
    radars = {'zhh14': {'name': 'Ku', 'title': 'Ku-band Reflectivity'},
              'zhh35': {'name': 'Ka', 'title': 'Ka-band Reflectivity'},
              'zhh95': {'name': 'W', 'title': 'W-band Reflectivity'},
              'ldrhh14': {'name': 'ldr', 'title': 'LDR at Ku-band '},
              'z95n': {'name': 'nadir', 'title': 'Nadir '},
              'vel14': {'name': 'vel', 'title': 'Ku-band Doppler Vel'}}#,
              # 'roll': {'name': 'roll', 'title': 'Left/Right Plane Roll'}}
              # 'isurf': {'name': 'isurf', 'title': 'index of best guess for location of surface'}}

    for i in radars.keys():
        if i in apr.keys():
            apr[i] = np.ma.masked_where((apr[i] <= -99) | (np.isnan(apr[i])), apr[i])
            apr[radars[i]['name']] = apr.pop(i)

    apr['DFR_1'] = apr['Ku'] - apr['Ka']  # Ku - Ka
    apr['time_gates'] = time_gate
    apr['time_dates'] = time_dates

    if flag == 0:
        apr['DFR_3'] = apr['Ka'] - apr['W']  # Ka - W
        apr['DFR_2'] = apr['Ku'] - apr['W']  # Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward]'
    else:
        apr['DFR_3'] = np.array([])  # Ka - W
        apr['DFR_2'] = np.array([])  # Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward], Note No W band avail'

    apr['ngates'] = apr['alt3D'].shape[0]

    _range = np.arange(15, apr['lat3D'].shape[0] * 30, 30)
    _range = np.asarray(_range, float)
    ind = np.where(_range >= apr['alt_nav'].mean())
    _range[ind] = np.nan
    apr['range'] = _range
    ds = xr.Dataset()
    for key, value in radars.items():
        if value['name'] in apr.keys():
            da = xr.DataArray(apr[value['name']],
                              dims={
                                  'range': np.arange(0, 550),
                                  'cross_track': np.arange(0, 24),
                                  'along_track': np.arange(apr[value['name']].shape[2])
                              },
                              coords={
                                  'lon3d': (['range', 'cross_track', 'along_track'], apr['lon3D']),
                                  'lat3d': (['range', 'cross_track', 'along_track'], apr['lat3D']),
                                  'time3d': (['range', 'cross_track', 'along_track'], apr['time_gates']),
                                  'alt3d': (['range', 'cross_track', 'along_track'], apr['alt3D'])})
            da.fillna(value=-9999)
            da.attrs['units'] = 'dBZ'
            da.attrs['standard_name'] = f"{value['title']}"
            ds[value['name']] = da
    return ds


def plane_loc(lat, lon, alt=None):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon.max() + 2, lon.min() - 2, lat.min() - 2, lat.max() + 2], crs=ccrs.Geodetic())
    ax.coastlines()

    # Draw the path of the fligt
    track = sgeom.LineString(zip(lon, lat))
    ax.add_geometries([track],
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='red',
                      linewidth=2)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    if alt:
        plt.pcolormesh(lon, lat, alt, transform=ccrs.PlateCarree())
    return fig, ax
    # plt.show()


def utils(file_obj):
    time = datetime.datetime(*[int(i) for i in file_obj['params_KUKA']['date_beg'][:]])
    return time


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    lat = []
    lon = []
    alt = []
    apr = apr3read(files[-2])
    plt.pcolormesh(apr.time3d[:, 12, :], apr.alt3d[:, 12, :], apr.Ku[:, 12, :],
                   vmin=0, vmax=40)
    plt.colorbar()
    plt.show()
    for file in files[-2:]:
        f = h5py.File(file, 'r')
        ds = f['hires']
        lat.append(ds['lat'][:][0])
        lon.append(ds['lon'][:][0])
        # alt.append(ds['alt3D'][:])

    lat = np.concatenate(lat).ravel()
    lon = np.concatenate(lon).ravel()
    # alt = np.concatenate(alt).ravel()
    fig, ax = plane_loc(lat=lat, lon=lon)
    # time = utils([files[-2]])
    # plt.title(f'{time}')
    plt.savefig('../results/track.png')
    plt.show()
    print(1)


if __name__ == '__main__':
    main()
