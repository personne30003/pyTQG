"""
Classe rassemblant les paramètres généraux de la simu. Gère aussi l'affichage de la sortie.
TODO : arreter d'initialiser les paramètres dans la classe, et les mettre dans un fichier JSON ('default.JSON'), avec leur documentation.
       Comme dans Fluid2d et pyRSW quoi.
"""
import os
import sys
import datetime
import Dealiasing
import copy()

class Params():
    def __init__(self, use_logger = False):
        
        self.Nx=100
        self.Ny=100
        self.x_bounds = (0.0, 2.0*np.pi)
        self.y_bounds = (-1.0, 1.0)
        self.list_geometry = ('basin', 'biperiodic', 'zonal_channel')
        self.geometry = 'biperiodic'

        self.dealias_params = Dealiasing.DealiasParams()
        
        self.time_scheme='Euler'
        self.max_time=1000.0
        self.max_it=1000000
        self.max_speed = 50.0#arbitraire, à modifier. Un dépassement de cette valeur engendre l'arret du programme
        
        self.output_path=os.getcwd()
        self.date = datetime.datetime.now()
        date_frm = self.date.strftime("%Y:%m:%d-%H:%M:%S")
        self.exp_name = f'Exp_{date_frm}'

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        pass