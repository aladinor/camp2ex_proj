#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
from re import split
import seaborn as sns
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']


def main():
    sensor = ['2DS10', 'Hawk2DS10']
    ls_lear = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
    lear_df = [pd.read_pickle(i) for i in ls_lear]
    ls_df = [i for i in lear_df if i.attrs['type'] in sensor]
    dates = ls_df[0].index.intersection(ls_df[1].index)
    ds10 = ls_df[0].loc[dates].filter(like='nsd')
    hawkds10 = ls_df[1].loc[dates].filter(like='nsd')
    diff = ds10 - hawkds10
    plt.figure(figsize=(25, 8))
    # ax = sns.scatterplot(x=ds10[ds10.columns[0]], y=hawkds10[hawkds10.columns[0]])
    ax1 = sns.boxplot(data=diff[diff.columns[15:35]])
    ax1.set(yscale="log")
    # ax1.set_xlim(-10e3, 10e3)
    # ax1.set_ylim(-10e3, 10e3)
    print(1)
    pass


if __name__ == '__main__':
    main()
