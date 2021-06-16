import h5py
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import glob
import numpy as np
import datetime


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    hdf_ds = h5py.File(files[-2], 'r')
    lores = hdf_ds['lores']['zhh14'][:]
    for i in range (lores.shape[0]):
        plt.close('all')
        fig, ax = plt.subplots()
        ax.pcolor(lores[i,:,:])
        plt.savefig(f'../results/lores_ref_ku_{i}')


if __name__ == '__main__':
    main()