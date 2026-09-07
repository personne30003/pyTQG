
"""
Résolution de l'équation de Burgers pour CLs périodiques
Possibilité de mettre du déaliasing (règle des 2/3)
"""
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
import IPython.display as ipd
import xrft

#Modules maison
import fourier as Fourier
import parameters as Params
import time_scheme as TimeScheme
import time

class Bu_Fourier:
    def __init__(self, Nx = 128, x_inter = (0.0, 2.0*np.pi), CFL_adv = 0.9, CFL_diff = .1):
        if len(x_inter) != 2:
            raise ValueError("x must be a 2 items tuple")
        if x_inter[0] >= x_inter[1]:
            raise ValueError("x[0] must be < x[1]")
        self.CFL_adv = CFL_adv
        self.CFL_diff = CFL_diff
        self.Nx = Nx
        self.intervalle_x = x_inter
        self.x = np.linspace(x_inter[0], x_inter[1], self.Nx, endpoint = True)
        self.dx = self.x[1] - self.x[0]
        self.kx = 2.0*np.pi*scipy.fft.fftfreq(self.Nx, d=self.dx)
    def Solve(self, 
              u_i,
              viscosite = 0.1,
              activate_dealias = False,
              coeff_dealias = 2.0/3.0,
              time_scheme = 'RK2',
              Tmax = 10.0,
              T_save = 0.1,
              u_max_stop = 50.0,
              N_it_max = 10000):

        def dealias(U, real = True):
            U_hat = scipy.fft.fft(U)
            U_hat_dealias = np.where(np.abs(self.kx) < coeff_dealias*np.abs(self.kx).max(), U_hat.copy(), 0.0)
            U_phys = scipy.fft.ifft(U_hat_dealias)
            return U_phys.real if real else U_phys
        
        def rhs_Bu_Fourier(uu, tt):
            u_du_dx = -uu*Fourier.Fourier_deriv(uu, order=1)
            if activate_dealias:
                u_du_dx = dealias(u_du_dx)
            res = -u_du_dx + viscosite*Fourier.Fourier_deriv(uu, order=2)
            return res

        def estimate_dt(max_U):
            dt_adv = self.CFL_adv*self.dx/max_U
            dt_visc = self.CFL_diff*(self.dx**2)/viscosite
            print(f"dt_adv = {dt_adv}")
            print(f"dt_diff = {dt_visc}")
            return min(dt_adv, dt_visc)
        params = Params.Params()
        params.time_scheme = time_scheme
        TS = TimeScheme.TimeScheme(params, rhs_Bu_Fourier, u_i.copy())
        print("#########################################################")
        print(f"Paramètres de la simu")
        print(f"CFL_adv = {self.CFL_adv}")
        print(f"CFL_diff = {self.CFL_diff}")
        print(f"viscosité = {viscosite}")
        print(f"dealiasing = {activate_dealias}")
        print(f"coeff_dealiasing = {coeff_dealias}")
        print(f"Nx = {self.Nx}")
        print(f"Intervalle = {self.intervalle_x}")
        comp_KE = lambda uu : Fourier.Fourier_quad(uu**2, self.dx)
        comp_u_max = lambda uu:np.abs(uu).max()
        
        liste_u = [u_i.copy()]
        liste_KE = [comp_KE(u_i)]

        t = 0.0
        liste_t = [t]

        dt = estimate_dt(comp_u_max(u_i))
        liste_dt = [dt]

        t_next_his = T_save
        N_it = 0
        while (t <= Tmax) : 
            ipd.clear_output(wait = True)
            N_it += 1
            t+=dt
            u_i= TS.Step(dt)
            
            Umax_i = comp_u_max(u_i)
            dt = estimate_dt(Umax_i)
            
            ke_i = comp_KE(u_i)
            if Umax_i >= u_max_stop:
                print(f"Dépassement de valeur limite (u_max = {Umax_i}) en t) {t}")
                break
            if dt <= 1.0e-16:
                print(f"dt trop faible (1e-16) à t = {t}. Stop")
                break
            if (t >= t_next_his):
                print("Sauvegarde des sorties")
                liste_dt.append(dt)
                liste_u.append(u_i)
                liste_t.append(t)
                liste_KE.append(ke_i)
                t_next_his += T_save
            
            print("####################################################")
            print(f"N_it = {N_it}")
            print(f"t= {t}")
            print(f"U_max = {Umax_i}")
            print(f"dt = {dt}")
            print(f"Energie cinétique totale = {ke_i:.4e}")
            print("####################################################")
            if N_it >= N_it_max : 
                print("nombre maximum d'itérations atteint")
                break
        print("####################################################")
        print(f"nombre d'itérations {N_it}") 
        Ds_res = xr.Dataset(data_vars = {'u':(['t', 'x'], liste_u),
                                       'ke':(['t'], liste_KE),
                                       'dt':(['t'], liste_dt)}, 
                          coords = {'t':('t', liste_t), 'x':('x',self.x)})
        Ds_res['spec_u'] = xrft.power_spectrum(Ds_res['u'], dim ='x')
        Ds_res.attrs['Number of points']= self.Nx
        Ds_res.attrs['viscosity'] = viscosite
        Ds_res.attrs['CFL_adv']=self.CFL_adv
        Ds_res.attrs['CFL_diff']= self.CFL_diff
        Ds_res.attrs['TimeScheme'] = params.time_scheme
        Ds_res.attrs['activate_dealias'] = activate_dealias
        Ds_res.attrs['coeff_dealias'] = coeff_dealias
        ipd.display(Ds_res)
        return Ds_res