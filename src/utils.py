#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from configparser import ConfigParser


def make_dir(path):
    """
    Makes directory based on path.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_pars_from_ini(file_name='../config/camp2ex.ini'):
    """
    Returns dictionary with data for creating an xarray dataset from hdf5 file
    :param file_name: configuration filename .ini file name
    :type file_name: str
    :return: data from config files
    """
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(file_name)

    dt_pars = {}

    groups = parser.sections()

    for group in groups:
        db = {}
        params = parser.items(group)

        for param in params:
            try:
                db[param[0]] = eval(param[1])

            except ValueError:
                db[param[0]] = param[1].strip()

            except NameError:
                db[param[0]] = param[1].strip()

            except SyntaxError:
                db[param[0]] = param[1].strip()

        dt_pars[group] = db

    return dt_pars


if __name__ == '__main__':
    pass
