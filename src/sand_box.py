import h5py
import matplotlib.pyplot as plt
import glob
from pyart.graph import cm
import datetime


def utils(file_obj):
    time = datetime.datetime(*[int(i) for i in file_obj['params_KUKA']['date_beg'][:]])
    return time


def main():
    path_file = '../data'
    files = glob.glob(f'{path_file}/*Wn.h5')
    files.sort()
    hdf_ds = h5py.File(files[-2], 'r')
    lores = hdf_ds['lores']['zhh35'][:]
    for i in range(lores.shape[0]):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 4))
        a = ax.imshow(lores[i, :, :], cmap='pyart_NWSRef', vmin=-10, vmax=35)
        ax.set_aspect(2)
        fig.colorbar(a, ax=ax, orientation='horizontal')
        plt.savefig(f'../results/Ka/ref/lores_ref_ka_{i}')


if __name__ == '__main__':
    main()