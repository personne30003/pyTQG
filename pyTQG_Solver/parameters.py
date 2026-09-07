"""
Classe rassemblant les paramètres généraux de la simu.
TODO : arreter d'initialiser les paramètres dans la classe, et les mettre dans un fichier JSON ('default.JSON'), avec leur documentation.
       Comme dans Fluid2d et pyRSW quoi.
"""
import os
import sys
import datetime
import dealiasing
import copy
import numpy as np
import dataclasses


class Params():
    def __init__(self):
        
        self.Nx=100
        self.Ny=100
        self.x_bounds = (0.0, 2.0*np.pi)
        self.y_bounds = (-1.0, 1.0)
        self.list_geometry = ('basin', 'biperiodic', 'zonal_channel')
        self.geometry = 'biperiodic'

        self.dealias_params = dealiasing.DealiasParams()
        
        self.time_scheme='Euler'
        self.max_time=10.0
        self.max_it=1000000
        self.max_speed = 50.0#arbitraire, à modifier. Un dépassement de cette valeur engendre l'arret du programme
        
        self.output_path=os.getcwd()
        date = datetime.datetime.now()
        date_frm = date.strftime("%Y:%m:%d-%H:%M:%S")
        self.date = date_frm
        
        self.exp_name = f'Exp_{date_frm}'
        self.exp_dir = os.path.dirname(os.getcwd())#On se place dans le dossier parent. C'est à dire le dossier juste avant le répertoire du module

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        str_out = "Params : \n"
        for attr, val in self.__dict__.items():
            str_out += f" - {attr} = {val}\n"
        return str_out

    def to_NETCDF_attrs(self):
        NC_attrs = {}
        dic_dealias = dataclasses.asdict(self.dealias_params)
        dic_dealias['apply_dealias'] = str(dic_dealias['apply_dealias'])
        for attr, val in self.__dict__.items():
            if val is None:
                NC_attrs[attr] = 'None'
            if attr == 'dealias_params':
                continue
            else:
                NC_attrs[attr] = val
        NC_attrs = {**NC_attrs, **dic_dealias}
        #print(NC_attrs)
        return NC_attrs
        