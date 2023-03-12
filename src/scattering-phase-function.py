#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import sys
import numpy as np
import dask
import matplotlib.pyplot as plt
from re import split
from pytmatrix import tmatrix, tmatrix_aux, refractive
from pytmatrix.scatter import sca_xsect
from labellines import labelLine, labelLines

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']


def _scatterer(d, ar, wlg, m, angle=180.0, rt=tmatrix.Scatterer.RADIUS_MAXIMUM):
    """
    Function that computes the scatterer object using pytmatrix package
    :param d: diameter in mm
    :param ar:  axis ratio
    :param wlg: wavelength in mm
    :param rt: maximum radius in mm
    :return: list of scatterers objects
    """
    return tmatrix.Scatterer(radius=d / 2., wavelength=wlg, m=m, axis_ratio=1.0 / ar,
                             phi=angle, radius_type=rt)


def get_phase(x):
    return x.get_Z()[0, 0]


def phase_dask(d, wl="Ku", temp=0, m=None):
    if wl == "Ku":
        wlg = tmatrix_aux.wl_Ku
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_Ku]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ku]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ku]
    elif wl == "Ka":
        wlg = tmatrix_aux.wl_Ka
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_Ka]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ka]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ka]
    elif wl == "W":
        wlg = tmatrix_aux.wl_W
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_W]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_W]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_W]
    else:
        raise Exception('wl {} not valid. Please use Ka, Ku, or W'.format(wl))
    #     phase = []
    #     for i in d:
    #         scat = [dask.delayed(_scatterer)(i, 1, wlg, m, j) for j in np.arange(0, 180, 1)]
    #         phase.append([dask.delayed(get_phase)(k) for k in scat])
    #     return np.array(dask.compute(*phase, scheduler="processes"))
    return np.array([[_scatterer(i, 1, wlg, m, j).get_Z()[0, 0] for j in np.arange(0, 180, 1)] for i in d])


def size_parm(d, wl):
    if wl == "Ku":
        wlg = tmatrix_aux.wl_Ku
    elif wl == "Ka":
        wlg = tmatrix_aux.wl_Ka
    elif wl == "W":
        wlg = tmatrix_aux.wl_W
    return 2 * np.pi * (d / 2) / wlg


def scatt_eff_dask(d, wl="Ku", temp=0, m=None):
    if wl == "Ku":
        wlg = tmatrix_aux.wl_Ku
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_Ku]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ku]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ku]
    elif wl == "Ka":
        wlg = tmatrix_aux.wl_Ka
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_Ka]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ka]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_Ka]
    elif wl == "W":
        wlg = tmatrix_aux.wl_W
        if m is None:
            if temp == 0:
                m = refractive.m_w_0C[tmatrix_aux.wl_W]
            elif temp == 10:
                m = refractive.m_w_10C[tmatrix_aux.wl_W]
            else:
                m = refractive.m_w_10C[tmatrix_aux.wl_W]
    else:
        raise Exception('wl {} not valid. Please use Ka, Ku, or W'.format(wl))
    x = size_parm(d=d, wl=wl)
    sigma = []
    for i in d:
        scat = dask.delayed(_scatterer)(i, 1, wlg, m)
        sig = dask.delayed(sca_xsect)(scat)
        sigma.append(sig)
    sigma_ = np.array(dask.compute(*sigma, scheduler="processes"))
    qs = sigma_ / (np.pi * (d / 2) ** 2)
    return qs, x, m


def get_radius(wl, x):
    if wl == "Ku":
        wlg = tmatrix_aux.wl_Ku
    elif wl == "Ka":
        wlg = tmatrix_aux.wl_Ka
    elif wl == "W":
        wlg = tmatrix_aux.wl_W
    return wlg * x / (2 * np.pi)


def main():
    # d = get_radius(wl="Ku", x=np.array([0.1, 0.3, 1, 3, 10, 30])) * 2
    x = np.array([0.1, 0.3, 1., 3., 10., 30.])
    d = get_radius(wl="Ku", x=x) * 2
    phase_ku = phase_dask(d, wl="Ku", m=complex(1.33, 0))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    for i in range(phase_ku.shape[0]):
        ax.plot(np.arange(1, 181, 1), phase_ku[i, :], c="k", lw=0.8, label=f'X={x[i]}')
    ax.set_ylabel(r'$P(\Theta)$')
    ax.set_xlabel(r'$\Theta$')
    ax.set_yscale('log')
    labelLines(fig.gca().get_lines(), color='k', fontsize=4, align=False)
    plt.show()
    print(1)


if __name__ == "__main__":
    main()