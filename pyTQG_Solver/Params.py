"""
Classe rassemblant les paramètres généraux de la simu
Code pour le moment tiré de l'UE "projet pro" (non abouti), donc à modifier selon les besoins
"""
import os


class Params():
    def __init__(self):
        self.Nx=100
        self.Ny=100
        self.dx=0.1
        self.dy=self.dx
        self.time_scheme='LeapFrog'
        self.max_time=1000.0
        self.max_it=1000
        self.out_path=os.getcwd()