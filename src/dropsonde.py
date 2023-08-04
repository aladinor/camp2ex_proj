#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import sys
import os
from re import split
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini
import fsspec

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def adjust_longitude(dataset: xr.Dataset) -> xr.Dataset:
    """Swaps longitude coordinates from range (0, 360) to (-180, 180)
    Args:
        dataset (xr.Dataset): xarray Dataset
    Returns:
        xr.Dataset: xarray Dataset with swapped longitude dimensions
    """
    lon_name = "lon"  # whatever name is in the data

    # Adjust lon values to make sure they are within (-180, 180)
    dataset["_longitude_adjusted"] = xr.where(
        dataset[lon_name] > 180, dataset[lon_name] - 360, dataset[lon_name]
    )
    dataset = (
        dataset.swap_dims({lon_name: "_longitude_adjusted"})
            .sel(**{"_longitude_adjusted": sorted(dataset._longitude_adjusted)})
            .drop(lon_name)
    )

    dataset = dataset.rename({"_longitude_adjusted": lon_name})
    return dataset


def get_temp_dew(lat, lon, time):
    url_temp = f"s3://era5-pds/zarr/{time:%Y}/{time:%m}/data/air_temperature_at_2_metres.zarr"
    url_dew = f"s3://era5-pds/zarr/{time:%Y}/{time:%m}/data/dew_point_temperature_at_2_metres.zarr"
    url_p = f"s3://era5-pds/zarr/{time:%Y}/{time:%m}/data/surface_air_pressure.zarr/"
    ds_temp = xr.open_zarr(fsspec.get_mapper(url_temp, anon=True), consolidated=True)
    ds_temp = adjust_longitude(ds_temp)
    tem = (ds_temp.sel(time0=f'{time:%Y%m%d %H%M}', lat=lat, lon=lon,
                       method='nearest').air_temperature_at_2_metres.values - 273) * units("degC")
    ds_dew = xr.open_zarr(fsspec.get_mapper(url_dew, anon=True), consolidated=True)
    ds_dew = adjust_longitude(ds_dew)
    dew = (ds_dew.sel(time0=f'{time:%Y%m%d %H%M}', lat=lat, lon=lon,
                      method='nearest').dew_point_temperature_at_2_metres.values - 273) * units("degC")
    ds_p = xr.open_zarr(fsspec.get_mapper(url_p, anon=True), consolidated=True)
    ds_p = adjust_longitude(ds_p)
    p = (ds_p.sel(time0=f'{time:%Y%m%d %H%M}', lat=lat, lon=lon,
                  method='nearest').surface_air_pressure.values) / 100 * units("hPa")
    lcl_pressure, lcl_temperature = mpcalc.lcl(p, tem, dew)
    cb = mpcalc.pressure_to_height_std(lcl_pressure)
    data = {'t_era': tem.m, 'td_era': dew.m, 'p_era': p.m, 'lcl_t': lcl_temperature.m,
            'lcl_p': lcl_pressure.m, 'cb_era': cb.m}
    print('done')
    return pd.DataFrame(data, index=[time])


def plot_sounding(df, save=False):
    p = df['pres'].values * units.hPa
    T = df['tdry'].values * units.degC
    Td = df['dp'].values * units.degC
    wind_speed = df['wspd'].values * units.knots
    wind_dir = df['wdir'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    fig = plt.figure(figsize=(9, 9))
    add_metpy_logo(fig, 115, 100)
    skew = SkewT(fig, rotation=45)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    skew.plot_barbs(p, u, v)
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 60)

    # Set some better labels than the default
    skew.ax.set_xlabel(f'Temperature ({T.units:~P})')
    skew.ax.set_ylabel(f'Pressure ({p.units:~P})')

    # Calculate LCL height and plot as black dot. Because `p`'s first value is
    # ~1000 mb and its last value is ~250 mb, the `0` index is selected for
    # `p`, `T`, and `Td` to lift the parcel from the surface. If `p` was inverted,
    # i.e. start from low value, 250 mb, to a high value, 1000 mb, the `-1` index
    # should be selected.
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    print(mpcalc.pressure_to_height_std(lcl_pressure))
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    skew.plot(p, prof, 'k', linewidth=2)

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, prof, Td)
    skew.shade_cape(p, T, prof)

    # An example of a slanted line at constant T -- in this case the 0
    # isotherm
    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    # Show the plot
    plt.show()
    if save:
        plt.savefig(f"../results/{pd.to_datetime(df.time.values[0]): }.pnd")


def ret_lcl(df):
    p = df['pres'].values * units.hPa
    T = df['tdry'].values * units.degC
    Td = df['dp'].values * units.degC
    try:
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        cb = mpcalc.pressure_to_height_std(lcl_pressure)
        series = df[['pres', 'lon', 'lat', 'gpsalt']].to_dataframe().iloc[1, :]
        series['cb'] = cb.m * 1000
        series['lcl_p'] = lcl_pressure.m
        series['lcl_t'] = lcl_temperature.m
        series = series.rename({'lat': 'lat_ds', 'lon': 'lon_ds'})
        return series
    except IndexError:
        pass


def main():
    snd_files = glob.glob(f"{path_data}/data/drop_sondes/*.nc")
    df_nn = pd.read_parquet(f"{path_data}/data/all_data.parquet").set_index('time')
    ser = []
    for file in snd_files:
        ds = xr.open_dataset(file)
        ds = ds.dropna('time')
        ser.append(ret_lcl(ds))
    df_cb = pd.concat(ser, axis=1).T
    df_all = pd.merge_asof(df_nn.sort_index(), df_cb.sort_index(), left_index=True,
                           right_index=True, direction="nearest")
    df_all['hacb'] = df_all['altitude'] - df_all['cb']
    df_all['time'] = df_all.index
    gp = df_all.groupby([df_all.time.dt.dayofyear, df_all.time.dt.hour])
    keys = list(gp.groups.keys())
    print(1)
    res = []
    for key in keys:
        _df = gp.get_group(key)
        res.append(get_temp_dew(lat=_df.iloc[0]['lat'], lon=_df.iloc[0]['lat'], time=_df.iloc[0]['time']))
    df_era = pd.concat(res)
    # df_era = df_era.drop(columns=['time'])
    df_all = pd.merge_asof(df_all, df_era, left_index=True, right_index=True, direction='nearest')
    df_all['cb_era'] = df_all['cb_era'] * 1000
    df_all['h_era'] = df_all['altitude'] - df_all['cb_era']
    df_all.to_parquet('../results/all_data_cb.parquet')
    #
    print(1)
    pass


if __name__ == '__main__':
    main()
    pass
