#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import urllib.request
import wget
import multiprocessing as mpc
from functools import partial


def download_data(url_paht, path):
    wget.download(url_paht, path)


def main():
    html_page = urllib.request.urlopen("https://www-air.larc.nasa.gov/cgi-bin/ArcView/camp2ex#TANELLI.SIMONE/")
    soup = BeautifulSoup(html_page, features="lxml")
    table = soup.find('table', attrs={'class': 'tablesorter', 'id': 'divPIContent_11'})
    links = [f"https://www-air.larc.nasa.gov{link.get('href')}" for link in table.findAll('a')]
    path = '/media/alfonso/drive/Alfonso/camp2ex_proj/data'
    pool = mpc.Pool()
    partial_funct = partial(download_data, path=path)
    pool.map(partial_funct, links[10:30])
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
