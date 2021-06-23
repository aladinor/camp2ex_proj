import h5py
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import glob
import numpy as np
import datetime


def apr3read(filename):
    """
    Function used for reading apr3 hdf file from CAMP2EX file using Randy's code
    (https://github.com/dopplerchase/Chase_et_al_2018/blob/master/apr3tocit_tools.py)
    :param filename: path and filename of the APR3 HDF file
    :return: dictionary with apr3 data
    """
    hdf = h5py.File(filename, "r")
    desired_data = ['alt3D', 'lat', 'lon', 'scantime', 'surface_index', 'isurf', 'alt_nav', 'zhh14', 'zhh35', 'ldrhh14',
                    'vel14', 'lon3D', 'lat3D', 'alt3D', 'z95n', 'roll', 'pitch', 'drift', 'z95s', 'z95n']
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
    radars = {'zhh14': 'Ku', 'zhh35': 'Ka', 'z95s': 'W', 'ldrhh14': 'ldr', 'z95n': 'Nadir'}
    for i in radars.keys():
        if i in apr.keys():
            apr[i] = np.ma.masked_where((apr[i] <= -99) | (np.isnan(apr[i])), apr[i])
            apr[radars[i]] = apr.pop(i)

    apr['DFR_1'] = apr['Ku'] - apr['Ka']  # Ku - Ka

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

    return apr


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
    apr = apr3read(files[0])
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
    plt.show()
    print(1)


if __name__ == '__main__':
    main()
