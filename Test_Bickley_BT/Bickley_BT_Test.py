######Juste pour les tests, après ce sera intégré dans l'environnement##########
from pathlib import Path
import sys

ROOT = Path.cwd().parent      # pyTQG
SRC = ROOT / "pyTQG_solver"

#print(ROOT)
#print(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
##################################################################################

import numpy as np
import os

import py_tqg
import parameters
import grid
import qg_barotropic
import init_fields

###########Paramètres généraux##############
params = parameters.Params()

params.exp_name = 'test_bickley'
params.exp_dir = os.getcwd()

params.Nx = 256
params.Ny = 256
params.x_bounds = (0.0, 2.0 * np.pi)
params.y_bounds = (-1.0, 1.0)
params.geometry = 'zonal_channel'

params.dealias_params.apply_dealias = False

params.max_speed = 25.0

params.time_scheme = 'RK2'
params.adaptable_dt = True
params.cfl = 1.0

params.max_time = 2.0
params.freq_his = 0.05
params.freq_diags = 0.05

############Grille##########################
Grid = grid.Grid(params)


###########Modèle###########################
QG_BT = qg_barotropic.BarotropicQG(params, Grid)

QG_BT.inv_Rd2 = 0.0#dynamique purement barotrope
QG_BT.T_0 = 0.0
QG_BT.beta = 0.0

#Jet de Bickley de largeur gamma et de vitesse max U0
#Rappel : U(y) = U0 * sech^2(y/gamma)
U_ini = init_fields.BickleyJet(Grid.Y, 0.5, U0 = 1.0)


#Initialisation du modèle#
rel_vort  = QG_BT.vort_from_U(U_ini)
#perturbation aléatoire
rel_vort = rel_vort + np.random.random_sample(Grid.shape)

QG_BT.PV_from_vort(rel_vort, assign = True)
QG_BT.psi_from_PV(assign = True)

###########Boucle principale################
PyTQG = py_tqg.pyTQG(params, Grid, QG_BT)
PyTQG.loop()