from fluid2d import Fluid2d
from param import Param
from grid import Grid
import numpy as np

""" 
Meme configuration que pour Test_Bickley_BT :
modèle QG barotrope : 
beta= 0.0
R_d infini

pv = delta psi - psi / Rd^2 + beta y 

U_O = sech^2((y-2)/0.2)

CLs :
-en x : périodique
-en y : psi(0) = 0, psi(Ly) = 0 (transport nul)

perturbation aléatoire


"""

param = Param('default.xml')
param.modelname = 'quasigeostrophic'
param.expname = 'f2d_bickley_w_transport_qg'
param.expdir = r'/mnt/c/Users/evanl/Documents/Cours M2 POC/StageLOPS/pyTQG/Test_Bickley_BT/'


# domain and resolution
param.nx = 512
param.ny = 512
param.npy = 1
param.Lx = 2.0*np.pi
param.Ly = 2.0
param.geometry = 'xchannel'

# time
param.tend = 15.0
param.cfl = 1.2
param.adaptable_dt = True
param.dt = 1.
param.dtmax = 100.

# discretization
param.order = 5


# output
param.var_to_save = ['psi', 'vorticity', 'pv']
param.list_diag = ['ke', 'pv', 'pv2']
param.freq_his = 0.1
param.freq_diag = 0.1
# plot
param.plot_var = 'vorticity'
param.freq_plot = 10
a = 0.5
param.cax = [-a, a]
param.plot_interactive = True
param.colorscheme = 'imposed'
param.generate_mp4 = False

# physics
param.beta = 0.0
param.Rd = 1.0e38#1/R_d^2 = 0.0
param.forcing = False
param.forcing_module = 'forcing'  # not yet implemented
param.noslip = False
param.diffusion = False
param.isisland = True


psi0 = 0.0  # transport moyen (nul ici)

grid = Grid(param)
nh = grid.nh

f2d = Fluid2d(param, grid)
model = f2d.model


def BickleyJet(y, gamma, U0=1.0, center = 0.0):
    "Jet de Bickley, de largeur gamma et de valeur max U0 (en y=0), centré en center"
    return U0*(np.cosh((y - center)/gamma))**(-2.0)

vort = model.var.get('pv')
#print(id(vort))
vort[:] = BickleyJet(grid.yr,
                     0.2,
                     U0 = 5.0,
                     center = param.Ly/2.0)
#print(id(vort))
vort += np.random.uniform(low = -0.1, high = 0.1, size = grid.yr.shape)

if grid.j0 == 0:
    msk = grid.msk.copy()*0
    msk[:nh, :] = 1
    idx = np.where(msk == 1)
    grid.island.add(idx, 0.0)

if grid.j0 == param.npy-1:
    msk = grid.msk.copy()*0
    msk[-nh:-1, :] = 1
    idx = np.where(msk == 1)
    grid.island.add(idx, psi0)

# desactive la condition de glissement
if grid.j0 == 0:
    grid.msknoslip[:nh, :] = 1
if grid.j0 == param.npy-1:
    grid.msknoslip[-nh:, :] = 1



model.set_psi_from_pv()

f2d.loop()
