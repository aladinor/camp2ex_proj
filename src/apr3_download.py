#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import urllib.request
import wget
import multiprocessing as mpc
from re import split
from bs4 import BeautifulSoup
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini, make_dir


def download_data(url_path, path):
    if not os.path.exists(path):
        make_dir('/'.join(path.split('/')[:-1]))
        wget.download(url_path, out=path)
    else:
        pass


def link_finder(url, path, ident):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, features="lxml")
    table = soup.find('table', attrs={'class': 'tablesorter', 'id': f'{ident}'})
    links = [f"https://www-air.larc.nasa.gov{link.get('href')}" for link in table.findAll('a')]
    ids = [f"{path}/{i.getText().split('-')[-1].split('_')[0]}/{i.getText()}" for i in table.findAll('a')]
    return links, ids


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
    webpage = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?P3B=1#LAWSON.PAUL/"
    # webpage = "https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex?LEARJET=1#LAWSON.PAUL/"
    sensor = webpage.split('#')[-1].replace('.', '_')[:-1]
    path_save = f'{path_data}/data/{sensor}'
    # id_table = 'divPIContent_0'
    id_table = 'divPIContent_7'
    links, ids = link_finder(url=webpage, path=path_save, ident=id_table)
    # download_data(links[0], ids[0])
    pool = mpc.Pool()
    pool.starmap(download_data, zip(links, ids))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
