#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from re import split
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini


def moment_nth(sr_nd, dict_diameters, moment):
    mn = sr_nd * dict_diameters['delta_diam'] * dict_diameters['diameters'] ** moment
    return mn.sum()


def get_date(_file):
    with open(_file, 'r') as f:
        lines = f.readlines()
        sizes = np.array([int(i[i.find('(') + 1: i.find(')')].replace(' ', '').split('-')[0])
                 for i in lines[:200] if i.startswith('cbin')])
        dsizes = sizes[1:] - sizes[:-1]
        dsizes = np.append(dsizes, dsizes[-1])
        bin_cent = (sizes[1:] - sizes[:-1]) / 2 + sizes[:-1]
        bin_cent = np.append(bin_cent, sizes[-1] + dsizes[-1])
        dt_sizes = {i: j for i, j in zip(bin_cent, dsizes)}
        try:
            df = pd.to_datetime(''.join(lines[6].split(',')[:3]), format='%Y%m%d', utc=True)
            return df, dt_sizes
        except ValueError:
            df = pd.to_datetime(''.join(lines[6].split(',')[:3]).replace(' ', ''), format='%Y%m%d', utc=True)
            return df, dt_sizes


def read_file(_file, _date, skipr=78):
    df_fcdp = pd.read_csv(_file, skiprows=skipr, header=0, na_values=-999)
    df_fcdp.index = df_fcdp.Time_Start.map(lambda x: _date + pd.to_timedelta(x, unit='seconds'))
    return df_fcdp


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    # Data
    _file = f'{path_data}/data/LAWSON_PAUL/FCDP/CAMP2Ex-FCDP_P3B_20190824_R1.ict'
    # _file = f'{path_data}/data/LAWSON_PAUL/Hawk2DS10/CAMP2Ex-Hawk2DS10_Learjet_20190907_R0_L1.ict'
    # _file = f'{path_data}/data/LAWSON_PAUL/Hawk2DS50/CAMP2Ex-Hawk2DS50_Learjet_20190907_R0_L1.ict'
    # _file = f'{path_data}/data/LAWSON_PAUL/HawkFCDP/CAMP2Ex-HawkFCDP_P3B_20191005_R1.ict'
    # _file = f'{path_data}/data/LAWSON_PAUL/HVPS/CAMP2Ex-HVPS_Learjet_20190907_R0_L1.ICT'

    _type = _file.split('/')[-1].split('-')[-1].split('_')[0]
    dt_info = get_pars_from_ini(campaign='psd_params')
    # _line = dt_info[_type]['line']
    _date, dt_sizes = get_date(_file)
    df_fcdp = read_file(_file, _date, skipr=dt_info[_type]['skip_row'])

    # Plotting PDS
    list_size = dt_info[_type]['list_size']
    values = df_fcdp[df_fcdp.conc > 10].filter(like='cbin').iloc[0]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x=list(dt_sizes.keys()), y=values, c='k', marker='*')
    ax.set_yscale('log')
    ax.set_ylabel('Number Concentration (#/L)')
    ax.set_xlabel('Diameter (um)')
    ax.set_xlim(0, 3000)
    ax.grid()
    plt.show()
    print(1)
    #

    pass


if __name__ == '__main__':
    main()
