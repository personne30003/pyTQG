"""
Classe rassemblant les paramètres généraux de la simu. Gère aussi l'affichage de la sortie.
TODO : arreter d'initialiser les paramètres dans la classe, et les mettre dans un fichier JSON ('default.JSON'), avec leur documentation.
       Comme dans Fluid2d et pyRSW quoi.
"""
import os
import sys
import datetime

class Params():
    def __init__(self, use_logger = False):
        self.Nx=100
        self.Ny=100
        self.time_scheme='Euler'
        self.max_time=1000.0
        self.max_it=1000000
        self.max_speed = 50.0#arbitraire, à modifier. Un dépassement de cette valeur engendre l'arret du programme
        self.list_configs = ('basin', 'biperiodic', 'channel')
        self.config = 'biperiodic'
        self.output_path=os.getcwd()
        self.date = datetime.datetime.now()
        date_frm = self.date.strftime("%Y:%m:%d-%H:%M:%S")
        self.exp_name = f'Exp_{date_frm}'


    def __str__(self):
        pass