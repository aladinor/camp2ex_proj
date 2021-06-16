import h5py
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import glob
import numpy as np
import datetime


def apr3read(filename):
    """
    ===========

    This is for reading in apr3 hdf (HDF5 updated 2/21/18) files from OLYMPEX and return them all in one dictionary

    ===========

    filename = filename of the apr3 file
    """

    apr = {}
    flag = 0

    # Radar varibles in hdf file found by hdf.datasets
    radar_freq = 'zhh14'  # Ku
    radar_freq2 = 'zhh35'  # Ka
    radar_freq3 = 'z95s'  # W
    radar_freq4 = 'ldrhh14'  # LDR
    vel_str = 'vel14'  # Doppler
    ##

    hdf = h5py.File(filename, "r")

    listofkeys = hdf['lores'].keys()
    alt = hdf['lores']['alt3D'][:]
    lat = hdf['lores']['lat'][:]
    lon = hdf['lores']['lon'][:]
    time = hdf['lores']['scantime'][:]
    surf = hdf['lores']['surface_index'][:]
    isurf = hdf['lores']['isurf'][:]
    plane = hdf['lores']['alt_nav'][:]
    radar = hdf['lores'][radar_freq][:]
    radar2 = hdf['lores'][radar_freq2][:]
    radar4 = hdf['lores'][radar_freq4][:]
    vel = hdf['lores']['vel14'][:]
    lon3d = hdf['lores']['lon3D'][:]
    lat3d = hdf['lores']['lat3D'][:]
    alt3d = hdf['lores']['alt3D'][:]

    # see if there is W band
    if radar_freq3 in listofkeys:
        if 'z95n' in listofkeys:
            radar_nadir = hdf['lores']['z95n']
            radar_scanning = hdf['lores'][radar_freq3]
            radar3 = radar_scanning
            # uncomment if you want high sensativty as nadir scan (WARNING, CALIBRATION)
            # radar3[:,12,:] = radar_nadir[:,12,:]
        else:
            radar3 = hdf['lores']['z95s']
            print('No vv, using hh')
    else:
        radar3 = np.ma.array([])
        flag = 1
        print('No W band')

    # convert time to datetimes
    time_dates = np.asarray([[datetime.datetime.utcfromtimestamp(time[i, j])
                             for j in range(time.shape[1])] for i in range(time.shape[0])])
    # Create a time at each gate (assuming it is the same down each ray, there is a better way to do this)
    time_gate = np.asarray([[[time_dates[i, j] for j in range(time_dates.shape[1])]
                            for i in range(time_dates.shape[0])] for k in range(lat3d.shape[0])])
    # Quality control (masked where invalid)
    radar = np.ma.masked_where(radar <= -99, radar)
    radar2 = np.ma.masked_where(radar2 <= -99, radar2)
    radar3 = np.ma.masked_where(radar3 <= -99, radar3)
    radar4 = np.ma.masked_where(radar4 <= -99, radar4)

    # Get rid of nans, the new HDF has builtin
    radar = np.ma.masked_where(np.isnan(radar), radar)
    radar2 = np.ma.masked_where(np.isnan(radar2), radar2)
    radar3 = np.ma.masked_where(np.isnan(radar3), radar3)
    radar4 = np.ma.masked_where(np.isnan(radar4), radar4)

    apr['Ku'] = radar
    apr['Ka'] = radar2
    apr['W'] = radar3
    apr['DFR_1'] = radar - radar2  # Ku - Ka

    if flag == 0:
        apr['DFR_3'] = radar2 - radar3  # Ka - W
        apr['DFR_2'] = radar - radar3  # Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward]'
    else:
        apr['DFR_3'] = np.array([])  # Ka - W
        apr['DFR_2'] = np.array([])  # Ku - W
        apr['info'] = 'The shape of these arrays are: Radar[Vertical gates,Time/DistanceForward], Note No W band avail'

    apr['ldr'] = radar4
    apr['vel'] = vel
    apr['lon'] = lon
    apr['lat'] = lat
    apr['alt_gate'] = alt3d
    apr['alt_plane'] = plane
    apr['surface'] = isurf
    apr['time'] = time
    apr['timedates'] = time_dates
    apr['time_gate'] = time_gate
    apr['lon_gate'] = lon3d
    apr['lat_gate'] = lat3d

    # fileheader = hdf.select('fileheader')
    roll = hdf['lores']['roll']
    pitch = hdf['lores']['pitch']
    drift = hdf['lores']['drift']

    ngates = alt.shape[0]

    apr['ngates'] = ngates
    apr['roll'] = roll
    apr['pitch'] = pitch
    apr['drift'] = drift

    _range = np.arange(15, lat3d.shape[0] * 30, 30)
    _range = np.asarray(_range, float)
    ind = np.where(_range >= plane.mean())
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
    plt.show()


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
    # dt_info = {i['results'][0]['name'].split('.')[1]: pd.DataFrame(i['results'][0]['values']) for i in queries if
    #            i['sample_size'] != 0}

    lat = np.concatenate(lat).ravel()
    lon = np.concatenate(lon).ravel()
    # alt = np.concatenate(alt).ravel()
    plane_loc(lat=lat, lon=lon) #, alt=alt)
    # elevation = ds['elevation'][:]
    # ncf = Dataset(file, diskless=True, persist=False)
    # nch = ncf.groups.get('hdf5-name')
    # xds = xr.open_dataset(xr.backends.NetCDF4DataStore(nch))
    print(1)


if __name__ == '__main__':
    main()
