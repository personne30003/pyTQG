"""
Tiré de mon solveur (en construction) QG barotrope de l'UE Projet Pro.

Attention, cette classe n'a pas encore été testée et validée.

Schémas implémentés (tous explicites) : Euler, Leapfrog (avec filtre d'asselin), RK4, Heun, RK3 SSP. 

A faire : implémenter RK2 (pas indispensable, mais bon).

Fortement inspiré de Fluid2d (auteur, G.Roullet, LOPS). Oui, meme le nom est identique
"""

import numpy as np
import Params

class TimeScheme:
    "Résout une équation de la forme du/dt=rhs(u, t, *args) par un schéma explicite"
    def __init__(self, params, rhs, u0):
        self.SchemeList=\
            {
            'Euler' : self.Euler,
            'Leapfrog' : self.Leapfrog,
            'Heun' : self.Heun,
            'RK3_SSP' : self.RK3_SSP,
            'RK4' : self.RK4
            }
        if params.time_scheme not in self.SchemeList.keys():
            raise ValueError(f"type must be in {self.SchemeList.keys()} (value {params.time_scheme})")
            
        self.scheme_type=params.time_scheme
        self.operator_scheme=self.SchemeList[self.scheme_type]
        self.u=np.copy(u0)
        self.rhs=rhs#Attention il s'agit d'une fonction
        self.t=0.0
        self.nb_it=0
        #paramètres pour schéma LeapFrog
        self.LeapFrog_init_scheme='Euler'
        self.LeapFrog_Asselin_coeff=0.05
        self.LeapFrog_u_nm1=np.copy(u)
        self.first_it = True
        
    def Step(self, dt, *args):
        self.nb_it+=1
        self.t+=dt
        return self.operator_scheme(dt,  *args)
        
    def Euler(self, dt, *args):
        self.u = self.u + dt * self.rhs(u,self.t, *args)
        return self.u
        
    def Leapfrog(self, dt,*args):
        "A terminer"
        u_n=np.copy(self.u)
        if self.first_it:
            self.u=self.SchemeList[self.LeapFrog_init_scheme](dt,self.t,*args)#u_1
            self.first_it = False
        else :
            #Pas valide si pas de temps non uniforme (c'est le cas dans mon solveur)
            self.u=self.LeapFrog_u_nm1+dt*self.rhs(self.LeapFrog_u_nm1, self.t-dt, *args)#u_n+1
            #Application d'un filtre d'Asselin
            u_n = u_n + self.LeapFrog_Asselin_coeff*(self.u + self.LeapFrog_u_nm1 - 2.0 * u_n)
        self.LeapFrog_u_nm1 = u_n
        return self.u

    def RK4(self, dt, *args):
        k1 = self.rhs(self.u, *args)
        k2 = self.rhs(self.u+0.5*dt*k1, *args)
        k3 = self.rhs(self.u+0.5*dt*k2, *args)
        k4 = self.rhs(self.u+dt*k3, *args)

        self.u = self.u + dt * (k1+2.0*k2+2.0*k3+k4)/6.0
        return self.u

    def Heun(self, dt,*args,t=0.0):
        u_1=self.u+dt*self.rhs(self.u, self.t, *args)
        self.u = self.u+dt/2.0*(self.rhs(u, self.t,*args)+self.rhs(u_1,self.t+dt, *args))
        return self.u
        
    def RK3_SSP(self, dt, *args,t=0.0):
        u1 = self.u+dt*self.rhs(self.u, self.t, *args)
        u2 = (3.0/4.0)*self.u+(1.0/4.0)*(u1+dt*self.rhs(u1, self.t, *args))
        self.u = (1.0/3.0)*self.u+(2.0/3.0)*(u2+dt*self.rhs(u2, self.t, *args))
        return self.u
