import glob
import sys
import os
from re import split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    ls_p3 = glob.glob(f'{path_data}/data/LAWSON_PAUL/P3B/*.pkl')
    ls_learjet = glob.glob(f'{path_data}/data/LAWSON_PAUL/Learjet/*.pkl')
    p3_df = [pd.read_pickle(i) for i in ls_p3]
    days_p3 = {i.attrs['type']: {'nrf': len(pd.Series(i.local_time).dt.floor('D').unique()),
                                 'dates': pd.Series(i.local_time).dt.floor('D').unique()} for i in p3_df}
    lear_df = [pd.read_pickle(i) for i in ls_learjet]
    days_lear = {i.attrs['type']: {'nrf': len(pd.Series(i.local_time).dt.floor('D').unique()),
                                   'dates': pd.Series(i.local_time).dt.floor('D').unique()} for i in lear_df}

    fcdp = [i for i in lear_df if (i.attrs['type'] == 'FCDP') | (i.attrs['type'] == 'HawkFCDP')]
    idx = fcdp[0].index.intersection(fcdp[1].index)
    df1 = fcdp[0].loc[idx].groupby(fcdp[0]['local_time'].dt.floor('D'))  # FCDP
    df2 = fcdp[1].loc[idx].groupby(fcdp[1]['local_time'].dt.floor('D'))  # HawkFCDP

    ncols = 3
    nrows = int(np.ceil(df1.ngroups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12), sharey=True, sharex=True)
    for (key, ax) in zip(df1.groups.keys(), axes.flatten()):
        for i in df1.get_group(key).filter(like='nsd', axis=1).columns[:10]:
            ax.scatter(df1.get_group(key)[i]/1000, df2.get_group(key)[i]/1000, s=0.5)
            ax.set_title(f'{key:%Y-%m-%d}')
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 300)
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, linewidth=0.5)
            # ax.set_aspect('equal')
    plt.xlabel('FCDP (#/L)')
    plt.ylabel('HawkFCDP (#/L)')
    plt.show()

    fcdp.filter(like='nsd', axis=1).groupby(fcdp['local_time'].dt.floor('D'))
    print(1)


if __name__ == '__main__':
    main()
    pass
