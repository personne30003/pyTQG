"""
Classe dédiée à l'enregistrement des variables de sorties dans des fichiers NETCDF4
Pour le moment, on n'inclut pas Model
"""

import xarray as xr
import Params
import State
import Grid
import os
import sys


class Output:
    def __init__(self, params, Grid, Model):
        pass
        