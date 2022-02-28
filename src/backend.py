#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
import sys
from re import split
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
p3_df = [pd.read_pickle(i) for i in ls_p3]
dt_day = [{'label': f'{i: %Y-%m-%d}', 'value': f'{i}'} for i in p3_df[0].local_time.dt.floor('D').unique()]
dt_sensor = [{"label": f"{i.attrs['type']}", "value": f"{i.attrs['type']}"} for i in p3_df]
dt_aircraft = [{"label": 'P3B', 'value': 'P3B'}, {"label": 'Learjet', 'value': 'Learjet'}]


def get_df(list_sensor):
    return [i for i in p3_df if i.attrs['type'] in list_sensor]


def main():
    df = get_df(['FCDP'])
    print(1)
    pass


if __name__ == '__main__':
    main()