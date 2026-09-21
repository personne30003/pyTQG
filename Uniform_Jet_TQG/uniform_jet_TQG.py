#####Juste pour les tests, après ce sera intégré dans l'environnement##########
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
import tqg_model
import init_fields

###########Paramètres généraux##############
params = parameters.Params()

params.exp_name = 'uniform_Jet_TQG'
params.exp_dir = os.getcwd()

params.Nx = 128
params.Ny = 128
params.x_bounds = (0.0, 2.0 * np.pi)
params.y_bounds = (0.0, 1.0)
params.geometry = 'zonal_channel'

params.dealias_params.apply_dealias = True
params.dealias_params.exp_alpha = 500.0
params.dealias_params.exp_p = 5.0
params.dealias_params.Fourier_coeff = 0.5

params.max_speed = 25.0

params.time_scheme = 'RK4'
params.adaptable_dt = True
params.cfl = 1.0

params.max_time = 15.0
params.freq_his = 0.05
params.freq_diags = 0.05




############Grille##########################
Grid = grid.Grid(params)


###########Modèle###########################
TQG = tqg_model.ThermalQG(params, Grid)

TQG.inv_Rd2 = 1.0
TQG.beta = 0.0

############Jet uniforme soumis à un gradient méridional de température
U0 = 1.0
U_ini = U0*np.ones(Grid.shape)
psi_ini = -U_ini*Grid.Y

TQG.T_0 = -U_ini * 1.0

alpha = -1.0
theta_ini = alpha * Grid.Y+2.0

############Réglage de la viscosité numérique################
TQG.visc2_q = 4.0e-9
TQG.visc2_theta = TQG.visc2_q
##############################################################


#Initialisation du modèle#
rel_vort  = TQG.vort_from_U(U_ini)
#perturbation aléatoire
rel_vort = rel_vort + np.random.uniform(low = -0.1, high = 0.1, size = Grid.shape)

TQG.PV_from_vort(vort = np.zeros_like(Grid.Y), theta = theta_ini, assign = True)
TQG.psi_from_PV(assign = True)

###########Boucle principale################
PyTQG = py_tqg.pyTQG(params, Grid, QG_BT)
PyTQG.loop()