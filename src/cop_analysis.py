import glob
import sys
import os
from re import split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir


def plot_scatter(df1, df2, fcdp, path_data, var):
    df1_gr = df1.groupby(df1['local_time'].dt.floor('D'))  # FCDP
    df2_gr = df2.groupby(df2['local_time'].dt.floor('D'))  # HawkFCDP
    ncols = 4
    nrows = int(np.ceil(df1_gr.ngroups / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharey=True, sharex=True)
    for (key, ax) in zip(df1_gr.groups.keys(), axes.flatten()):
        for i in df1_gr.get_group(key).filter(like=var, axis=1).columns[:10]:
            ax.scatter(df1_gr.get_group(key)[i]/1000, df2_gr.get_group(key)[i]/1000, s=0.5)
            ax.set_title(f'{key:%Y-%m-%d}')
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, linewidth=0.5)
    fig.supxlabel('FCDP (#/cc)')
    fig.supylabel('HawkFCDP (#/cc)')
    fig.tight_layout()
    save = f"{path_data}/results/LAWSON.PAUL/{fcdp[0].attrs['aircraft']}/{fcdp[0].attrs['type']}"
    make_dir(save)
    # fig.savefig(f"{save}/{fcdp[0].attrs['type']}_{var}.png")
    plt.show()


def plot_nsd(lear_df, _idx):
    fig, ax = plt.subplots()
    for i in lear_df[:-1]:
        x = i.attrs['sizes'][1:]
        y = i[i.index == _idx].filter(like='nsd').iloc[0].values
        y = np.where(y > 0, y, np.nan)
        ax.step(x, y, label=i.attrs['type'])
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.grid(which='both')
    ax.set_title(f"{_idx: %Y-%m-%d %H:%M:%S} (UTC) - {i.attrs['aircraft']}")
    ax.set_xlabel(f"$Diameter \ (\mu m)$")
    ax.set_ylabel("$Concentration \ (\#  L^{-1} \mu m^{-1})$")
    plt.show()



def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    ls_p3 = glob.glob(f'{path_data}/data/LAWSON.PAUL/P3B/all/*.pkl')
    ls_learjet = glob.glob(f'{path_data}/data/LAWSON.PAUL/LEARJET/all/*.pkl')
    p3_df = [pd.read_pickle(i) for i in ls_p3]
    days_p3 = {i.attrs['type']: {'nrf': len(pd.Series(i.local_time).dt.floor('D').unique()),
                                 'dates': pd.Series(i.local_time).dt.floor('D').unique()} for i in p3_df}
    lear_df = [pd.read_pickle(i) for i in ls_learjet]
    days_lear = {i.attrs['type']: {'nrf': len(pd.Series(i.local_time).dt.floor('D').unique()),
                                   'dates': pd.Series(i.local_time).dt.floor('D').unique()} for i in lear_df}
    fcdp = [i for i in lear_df if (i.attrs['type'] == 'FCDP') | (i.attrs['type'] == 'HawkFCDP') |
            (i.attrs['type'] == 'Page0')]
    df1 = pd.merge(fcdp[0], fcdp[2][['Temp']], left_index=True, right_index=True)
    df2 = pd.merge(fcdp[1], fcdp[2][['Temp']], left_index=True, right_index=True)
    temp = None
    # temp = 2
    # var = 'conc'
    var = 'nsd'
    if temp:
        df1 = df1[df1.Temp >= temp]
        df2 = df2[df2.Temp >= temp]
    idx = df1.index.intersection(df2.index)
    df1 = df1.loc[idx]
    df2 = df2.loc[idx]
    # plot_scatter(df1, df2, fcdp, path_data, var)
    _idx = lear_df[0][lear_df[0]['conc'] > 1000000].filter(like='nsd').index[0]
    plot_nsd(lear_df, _idx)
    print(1)


if __name__ == '__main__':
    main()
    pass
