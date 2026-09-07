"""
Classe dédiée à l'enregistrement des variables de sorties dans des fichiers NETCDF4:
- '[expname]_his.nc", qui stocke les champs (dépendant de x, y, t)
- '[expname]_diag.nc, qui stocke les nombres (ne dépendant que de t)
champs
"""


import os
#import sys
import pathlib
import netCDF4 as nc4

import warnings

import parameters
import QG_model
import grid
class Output:
    def __init__(self,
                 params : parameters.Params,
                 model : QG_model.QG_model,
                 grid : grid.Grid,
                 path_output : pathlib.Path):
        self._exp_dir = params.exp_dir
        self._exp_name = params.exp_name
        attrs_params = params.to_NETCDF_attrs()
        attrs_model = model.to_NETCDF_attrs()
        self.x = grid.x
        self.y = grid.y
        self.attrs_Ds = {**attrs_params, **attrs_model}
        self.name_his = self._exp_name + '_his.nc'
        self.name_diag = self._exp_name + '_diag.nc'
        self.path_his = str(path_output / self.name_his)
        self.path_diag = str(path_output / self.name_diag)
        #teste l'existence des fichiers
        if os.path.isfile(self.path_his) :
            warnings.warn(f"file {self.path_his} already exist. It will be deleted and recreated",
                          RuntimeWarning)
            os.remove(self.path_his)
        if os.path.isfile(self.path_diag):
            warnings.warn(f"file {self.path_diag} already exist. It will be deleted and recreated",
                          RuntimeWarning)
            os.remove(self.path_diag)

    def save_diags(self, state, t):
        pass

    def save_his(self, state, t):
        with nc4.Dataset(self.path_his, mode = 'a') as NC_file:
            NC_file.variables['t'][-1] =

    def __create_NC_fields(self, model : Model.Model):
        with nc4.Dataset(self.path_his, mode = 'w', format = 'NETCDF4') as NC_file:
            #Creation dimension
            NC_file.createDimension('x', self.x.size)
            NC_file.createDimension('y', self.y.size)
            NC_file.createDimension('time', None)

            #Creation coordonnées
            x_var  = NC_file.createVariable('x', 'f8', ('x',))
            y_var = NC_file.createVariable('y', 'f8', ('y',))
            NC_file.createVariable('t', 'f8', ('t',))

            x_var[:] = self.x
            y_var[:] = self.y

            for field in model.State.fields_vars :
                NC_file.createVariable(field, 'f8', ('t', 'x', 'y'), zlib = True)
            NC_file.title = self._exp_name + '_fields'

    def __create_NC_scalars(self, model : Model.Model):
        with nc4.Dataset(self.path_his, mode = 'w', format = 'NETCDF4') as NC_file:
            NC_file.createDimension('t', None)
            NC_file.createVariable('t', 'f8', ('t', ))

            for scalar in model.State.scalar_vars:
                NC_file.createVariable(scalar, 'f8', ('t'), zlib = True)
            NC_file.title = self._exp_name + '_diagnostics'
