"""
Classe dédiée à l'enregistrement des variables de sorties dans des fichiers NETCDF4:
- '[expname]_his.nc", qui stocke les champs (dépendant de x, y, t)
- '[expname]_diag.nc, qui stocke les nombres (ne dépendant que de t)
champs
"""

import xarray as xr
import Params
import Grid
import Model

import os
import sys
from pathlib import Path

import warnings

class Output:
    def __init__(self, params : Params.Params, grid : Grid.Grid, model : Model.Model, path_output : Path.Path = None ):
        self._exp_dir = params.exp_dir
        attrs_params = params.to_NETCDF_attrs()
        attrs_model = model.to_NETCDF_attrs()
        attrs_Ds = {**attrs_params, **attrs_model}

        self.name_his = params.exp_name + '_his.nc'
        self.name_diag = params.exp_name + '_diag.nc'

    def save_diags(self, state):
        pass

    def save_his(self, state):
        pass