from pathlib import Path
import sys

ROOT = Path.cwd().parent      # pyTQG
SRC = ROOT / "pyTQG_solver"

print(ROOT)
print(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import xarray as xr
import sympy as sp
import scipy
import os
#Modules maison
import Fourier
import Chebyshev
import Params
import TimeScheme
import Stab_Bu
import time

repertoire_courant = os.getcwd()
print(f"repertoire courant {repertoire_courant}")
expname = 'Burgers_Fourier'

x, nu, t, u_ana = sp.symbols('x nu t u_{ana}')
phi = (sp.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sp.exp(-(x - 4 * t - 2 * sp.pi)**2 / (4 * nu * (t + 1))))

u_ana = -2*nu*phi.diff(x)/phi+4
func_u_ana = sp.utilities.lambdify((x, t, nu), u_ana)

visc=0.0001

Nx = 128
x_lin = np.linspace(0.0, 2.0*np.pi,Nx, endpoint = False)
dx = x_lin[1]-x_lin[0]


rhs_Bu_Fourier = lambda uu, tt : -uu*Fourier.Fourier_deriv(uu, order=1) + visc*Fourier.Fourier_deriv(uu, order=2)
u_F = func_u_ana(x_lin, 0, visc)


Tmax = 10.0
CFL_adv = 0.9

N_it_max = 10000000
N_it = 0
u_max = 50.0


print(f"dx = {dx}")
print(f"visc = {visc}")


liste_t = [0.0]
liste_u = [u_F]
liste_KE = [Fourier.Fourier_quad(u_F**2, dx)]


t = 0.0
N_save_output = 1000
dt = CFL * dx/np.max(u_F)
liste_dt = [np.NaN]

params = Params.Params()
params.time_scheme = 'LeapFrog'
TS = TimeScheme.TimeScheme(params, rhs_Bu_Fourier, u_F.copy())
print("Calcul du contour de la zone de stabilité du schéma temporel")
Stab_Analyser = Stab_Bu.StabilityAnalysis(TS)
print("Fait")
kx = 2.0*np.pi*scipy.fft.fftfreq(Nx, d = dx)
VP_dx_Fourier = 1.0j*kx
VP_dx2_Fourier = -kx**2

liste_stab_ana = Stab_Analyser.Compute_stab_operators([VP_dx_Fourier, VP_dx2_Fourier])

stab_F_dx = liste_stab_ana[0]
stab_F_dx2 = liste_stab_ana[1]
comp_dt_adv = lambda umax, res_stab_ana: res_stab_ana.coeff_corr*res_stab_ana.d_min_stab/(res_stab_ana.max_abs_eigs*umax)
comp_dt_diff = lambda nu, res_stab_ana: res_stab_ana.coeff_corr*res_stab_ana.d_min_stab/(res_stab_ana.max_abs_eigs*visc)
print(f"methode d'intégration temporelle = {params.time_scheme}")
print(f"stab_F_dx = {stab_F_dx}")
print(f"stab_F_dx = {stab_F_dx2}")
print(f"kx = {kx}")
print(f"Schéma temporel {params.time_scheme}")
t0 = time.time()

def estimate_dt(max_U, Visc):
    dt_adv = comp_dt_adv(max_U, stab_F_dx)
    dt_visc = comp_dt_diff(Visc, stab_F_dx2)
    print(f"dt_adv = {dt_adv}")
    print(f"dt_diff = {dt_visc}")
    return min(dt_adv, dt_visc)
dt = estimate_dt(np.abs(u_F).max(), visc)
liste_dt= [dt]

print("debut boucle")
print(f"####################################################")
while (t <= Tmax) : 
    N_it += 1
    #dt = 0.0001
    t+=dt
    u_i= TS.Step(dt)
    #print(f"t={t}")
    #print(f"U max = {np.max(u_i):.3f}")
    #print(f"dt={dt:.3f}")
    Umax_i = np.abs(u_i).max()

    dt = estimate_dt(Umax_i, visc)
    if np.max(u_i) >= u_max:
        print(f"Dépassement de valeur limite (u_max = {u_max}) en t) {t}")
        break
    if dt <= 1.0e-16:
        print(f"dt trop faible (1e-16) à t = {t}. Stop")
        break
    if (N_it % N_save_output) == 0:
        print("Sauvegarde des sorties")
        liste_dt.append(dt)
        liste_u.append(u_i)
        liste_t.append(t)
        liste_KE.append(Fourier.Fourier_quad(u_i**2, dx))
    print("####################################################")
    print(f"N_it = {N_it}")
    print(f"t= {t}")
    print(f"U_max = {Umax_i}")
    print(f"dt = {dt}")
    print(f"Energie cinétique totale = {Fourier.Fourier_quad(u_i**2, dx):.4e}")
    print("####################################################")
    if N_it >= N_it_max : 
        print("nombre maximum d'itérations atteint")
        break
print("####################################################")
print(f"Fait en {(time.time()-t0)/60} minutes")
liste_u = np.array(liste_u)
print(f"nombre d'itérations {N_it}")




Ds_F = xr.Dataset(data_vars = {'u':(['t', 'x'], liste_u),
                               'ke':(['t'], liste_KE),
                               'dt':(['t'], liste_dt)}, 
                               coords = {'t':('t', liste_t), 'x':('x',x_lin)})
Ds_F.attrs['Number of points']= Nx
Ds_F.attrs['viscosity'] = visc

print(f"sauvegarde dans un fichier netcdf ({expname}.nc)")
Ds_F.to_netcdf(repertoire_courant + fr"\\{expname}.nc")
print("#####################################################")
print("###################Fin du programme##################")