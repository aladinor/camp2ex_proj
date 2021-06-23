import h5py
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import glob
from pyart.graph import cm
import numpy as np
import datetime


def utils(file_obj):
    time = datetime.datetime(*[int(i) for i in file_obj['params_KUKA']['date_beg'][:]])
    return time


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    hdf_ds = h5py.File(files[-2], 'r')
    lores = hdf_ds['lores']['vel14'][:]
    for i in range(lores.shape[0]):
        plt.close('all')
        fig, ax = plt.subplots()
        ax.pcolor(lores[i, :, :], cmap=cm.NWSRef)
        plt.savefig(f'../results/Ku/vel/lores_vel_ku_{i}')


if __name__ == '__main__':
    main()