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
import qg_model
import grid

class Output:
    def __init__(self,
                 params : parameters.Params,
                 model : qg_model.QG_model,
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
        self.name_diag = self._exp_name + '_diags.nc'
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
        self.__create_NC_fields(model)
        self.__create_NC_scalars(model)
        self.idx_diag = 0
        self.idx_his = 0
        self.__Model = model
        #self.Model = model#Juste pour debug

    def save_diags(self, t):
        with nc4.Dataset(self.path_diag, mode='a') as NC_file:
            NC_file.variables['t'][self.idx_diag] = t
            for scalar_var in self.__Model.State.scalar_vars:
                NC_file.variables[scalar_var][self.idx_diag] = self.__Model.State.scalar_values[scalar_var]
            NC_file.sync()
        self.idx_diag += 1

    def save_his(self, t):
        with nc4.Dataset(self.path_his, mode = 'a') as NC_file:
            NC_file.variables['t'][self.idx_his] = t
            for field_var in self.__Model.State.fields_vars :
                NC_file.variables[field_var][self.idx_his, :, :] = self.__Model.State.fields_values[field_var]
            NC_file.sync()
        self.idx_his +=1


    def __create_NC_fields(self, model : qg_model.QG_model):
        with nc4.Dataset(self.path_his, mode = 'w', format = 'NETCDF4') as NC_file:
            #Creation dimension
            NC_file.createDimension('x', self.x.size)
            NC_file.createDimension('y', self.y.size)
            NC_file.createDimension('t', None)

            #Creation coordonnées
            x_var  = NC_file.createVariable('x', 'f8', ('x',))
            y_var = NC_file.createVariable('y', 'f8', ('y',))
            NC_file.createVariable('t', 'f8', ('t',))

            x_var[:] = self.x
            y_var[:] = self.y

            for field in model.State.fields_vars :
                NC_file.createVariable(field, 'f8', ('t', 'x', 'y'), zlib = True)
            NC_file.title = self.name_his
            self.__write_nc_attrs(NC_file, self.attrs_Ds)
            NC_file.sync()

    def __create_NC_scalars(self, model : qg_model.QG_model):
        with nc4.Dataset(self.path_diag, mode = 'w', format = 'NETCDF4') as NC_file:
            NC_file.createDimension('t', None)
            NC_file.createVariable('t', 'f8', ('t', ))

            for scalar in model.State.scalar_vars:
                NC_file.createVariable(scalar, 'f8', ('t'), zlib = True)
            NC_file.title = self.name_diag
            self.__write_nc_attrs(NC_file, self.attrs_Ds)
            NC_file.sync()

    def __write_nc_attrs(self,nc_file : nc4.Dataset, attrs : dict):
        for key, value in attrs.items():
            nc_file.setncattr(key, value)
