#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import urllib.request
import wget
import multiprocessing as mpc
from re import split, compile
from bs4 import BeautifulSoup

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir


def download_data(url_path, path):
    if not os.path.exists(path):
        make_dir('/'.join(path.split('/')[:-1]))
        wget.download(url_path, out=path)
    else:
        pass


def link_finder(url, path, pi, sensor):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, features="lxml")
    root = f'XFS/CAMP2EX/2019/{sensor}_AIRCRAFT/{pi}/'
    if sensor == 'MERGE':
        sensor = 'MERGES'
        root = f'XFS/CAMP2EX/2019/{sensor}/{pi}/'
    table = [i for i in soup.findAll('table', attrs={'class': 'tablesorter'})
             if i.findAll('input')[0].get('id').startswith(root)][0]
    links = [f"https://www-air.larc.nasa.gov{link.get('href')}" for link in table.findAll('a')]
    ids = [f"{path}/{i.getText().split('-')[-1].split('_')[0]}/{i.getText()}" for i in table.findAll('a')]
    return links, ids


def download_camp2ex(webpage):
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    sensor = webpage.split('?')[-1].split('=')[0]
    pi = webpage.split('#')[-1][:-1]
    path_save = f'{path_data}/data/{pi}/{sensor}'
    links, ids = link_finder(url=webpage, path=path_save, pi=pi, sensor=sensor)
    # download_data(links[0], ids[0])
    pool = mpc.Pool()
    pool.starmap(download_data, zip(links, ids))
    pool.close()
    pool.join()


def main():
    webpage = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?P3B=1#LAWSON.PAUL/"
    webpage1 = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?LEARJET=1#LAWSON.PAUL/"
    # webpage2 = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?P3B=1#TANELLI.SIMONE/"
    webpage3 = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?MERGE=1#01_SECOND.P3B_MRG/"
    # ls_web = [webpage, webpage1]  # , webpage2]
    ls_web = [webpage3]
    for i in ls_web:
        download_camp2ex(i)


if __name__ == '__main__':
    main()
