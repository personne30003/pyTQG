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
expname = 'Burgers_Cheb_CFL'

visc=0.1

x_inf = 0.0
x_sup = np.pi


Nx_C = 128
x_C = Chebyshev.collocation_points(Nx_C, a=x_inf, b= x_sup)
dx_C = (x_sup - x_inf)*0.25*(np.pi/Nx_C)**2
D =Chebyshev.Cheb_mat(Nx_C, a=x_inf, b=x_sup, Dirichlet_BC = True)
D2 =Chebyshev.Cheb_mat(Nx_C, a=x_inf, b=x_sup,M=2, Dirichlet_BC = True)

def RHS_Bu_C(UU, tt):
    res = -UU*(D@UU) + visc*(D2@UU)
    return res


Tmax = 10.0
N_it_max = 10000000
N_it = 0
u_max = 50.0


print(f"dx = {dx_C}")
print(f"visc = {visc}")


CFL_adv = 0.9
CFL_diff = 0.1


liste_t = [0.0]

u_i = np.sin(x_C)
liste_u = [u_i]
liste_KE = [Chebyshev.Cheb_quad(u_i**2, a=x_inf, b=x_sup)]


t = 0.0
N_save_output = 1000

params = Params.Params()
params.time_scheme = 'RK2'
TS = TimeScheme.TimeScheme(params, RHS_Bu_C, u_i.copy())




print(f"methode d'intégration temporelle = {params.time_scheme}")
print(f"Schéma temporel {params.time_scheme}")
t0 = time.time()

def estimate_dt(max_U, Visc):
    dt_adv = CFL_adv*dx_C/max_U
    dt_visc = CFL_diff*(dx_C**2)/Visc
    print(f"dt_adv = {dt_adv}")
    print(f"dt_diff = {dt_visc}")
    return min(dt_adv, dt_visc)
dt = estimate_dt(np.abs(u_i).max(), visc)
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
    
    KE_i = Chebyshev.Cheb_quad(u_i**2, a=x_inf, b=x_sup)
    dt = estimate_dt(Umax_i, visc)
    if np.max(u_i) >= u_max:
        print(f"Dépassement de valeur limite (u_max >= {u_max}) en t) {t}")
        break
    if dt <= 1.0e-16:
        print(f"dt trop faible (1e-16) à t = {t}. Stop")
        break
    if (N_it % N_save_output) == 0:
        print("Sauvegarde des sorties")
        liste_dt.append(dt)
        liste_u.append(u_i)
        liste_t.append(t)
        liste_KE.append(KE_i)
    print("####################################################")
    print(f"N_it = {N_it}")
    print(f"t= {t}")
    print(f"U_max = {Umax_i}")
    print(f"dt = {dt}")
    print(f"Energie cinétique totale = {KE_i:.4e}")
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
                               coords = {'t':liste_t, 'x':x_C})

Ds_F.attrs['Number of grid points']= Nx_C
Ds_F.attrs['viscosity'] = visc
Ds_F.attrs['TimeScheme'] = params.time_scheme
Ds_F.attrs['CFL_adv'] = CFL_adv
Ds_F.attrs['CFL_diff'] = CFL_diff


print(f"sauvegarde dans un fichier netcdf ({expname}.nc)")
Ds_F.to_netcdf(repertoire_courant + fr"\\{expname}.nc")
print("#####################################################")
print("###################Fin du programme##################")