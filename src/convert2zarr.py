#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from re import split
import sys
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini
from src.apr3 import hdf2zar


def main():
    location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
    path_data = get_pars_from_ini(campaign='loc')[location]['path_data']
#    hdf2zar(path_data)
    print(path_data)
    pass


if __name__ == '__main__':
    main()

