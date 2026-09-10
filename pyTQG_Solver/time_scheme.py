"""
Tiré de mon solveur (en construction) QG barotrope de l'UE Projet Pro.

Schémas implémentés (tous explicites) : Euler, Leapfrog (avec filtre d'Asselin), RK4, Heun, RK2

Peut s'appliquer à tout objet (scalaire, array, DataArray ...) muni des opérateurs d'addition, multiplication/division
 par un scalaire et copie (obj.copy())

Fortement inspiré de Fluid2d (auteur : G.Roullet, LOPS). Oui, meme le nom est identique.

Note :
u0 est le array/State/DataArray  initial. Dans le cas où copy_initial = False, TimeScheme agit directement
sur l'objet (puisqu'ils partagent la meme adresse mémoire. Ainsi, on peut directement avoir sa valeur incrémentée,
sans avoir à faire (par exemple) TimeScheme.u. Pour cela, l'addition est effectuée via l'opérateur += (__iadd__).
Comme ça, l'adresse mémoire de l'objet initial reste la meme au cours du temps, ce que ne permets pas l'opérateur
d'addition simple : si on fait a = np.arange(10) ; a = a + 1, a l'opérateur __add__
créé un nouvel objet a à une adresse mémoire différente de l'objet initial.

Du coup les implémentations deviennent moins explicites ...

"""

import parameters
import numpy as np


class TimeScheme:
    "Résout une équation de la forme du/dt=rhs(u, t, *args) par un schéma explicite"
    def __init__(self, params : parameters.Params, rhs, u0, copy_initial = True):
        self.__SchemeList = {
            'Euler' : self.Euler,
            'LeapFrog' : self.LeapFrog,
            'Heun' : self.Heun,
            'RK2' : self.RK2,
            'RK4' : self.RK4
        }


        if params.time_scheme not in self.__SchemeList.keys():
            raise ValueError(f"type must be in {self.__SchemeList.keys()} (value {params.time_scheme})")
            
        self.scheme_type = params.time_scheme
        self.operator_scheme = self.__SchemeList[self.scheme_type]

        if not np.isscalar(u0):
            self.u=u0.copy() if copy_initial else u0
            self.LeapFrog_u_old = u0.copy()
            self.LeapFrog_u_f = u0.copy()
        else:
            self.u = u0
            self.LeapFrog_u_old = np.copy(u0)
            self.LeapFrog_u_f = np.copy(u0)
        #print(f"initialisation u={self.u}")
        self.rhs = rhs#Attention il s'agit d'une fonction
        self.t = 0.0
        #paramètres pour schéma LeapFrog
        self.LeapFrog_init_scheme = 'Euler'
        self.LeapFrog_Asselin_coeff = 0.05
        self.LeapFrog_dt_old = None
        self.first_it = True
        
    def Step(self, dt, *args):
        self.t += dt
        self.operator_scheme(dt,  *args)
        self.LeapFrog_dt_old = dt
        return self.u
        
    def Euler(self, dt, *args):
        self.u += dt * self.rhs(self.u, self.t, *args)
        #return self.u
        
    def LeapFrog(self, dt,*args):
        "Attention quand copy_initial = False, ici nous faisons plein de copies"
        if self.first_it:
            #print("first it")
            self.__SchemeList[self.LeapFrog_init_scheme](dt,*args)#u_1
            #self.RK2(dt, *args)
            self.first_it = False
        else :
            #Pas valide si pas de temps non uniforme (c'est le cas dans mon solveur)
            u_new=self.LeapFrog_u_old + 2.0*dt*self.rhs(self.u, self.t - self.LeapFrog_dt_old, *args)#u_n+1
            
            #Application d'un filtre d'Asselin
            self.LeapFrog_u_f = self.u + self.LeapFrog_Asselin_coeff*(u_new + self.LeapFrog_u_old - 2.0 * self.u)
            self.LeapFrog_u_old = self.LeapFrog_u_f.copy()
            self.u = u_new.copy()
        #self.LeapFrog_u_nm1 = u_n
        #return self.u
     
    def RK2(self, dt, *args) : 
        k1 = self.rhs(self.u, self.t, *args)
        k2 = self.rhs(self.u + dt * k1, self.t+dt)
        self.u += 0.5*dt*(k1+k2)
        
    def RK4(self, dt, *args):
        k1 = self.rhs(self.u, self.t, *args)
        k2 = self.rhs(self.u+0.5*dt*k1, self.t+dt/2.0, *args)
        k3 = self.rhs(self.u+0.5*dt*k2, self.t +dt/2.0, *args)
        k4 = self.rhs(self.u+dt*k3, self.t+dt, *args)

        self.u += dt * (k1+2.0*k2+2.0*k3+k4)/6.0
        #return self.u

    def Heun(self, dt,*args,t=0.0):
        u_1=self.u+dt*self.rhs(self.u, self.t, *args)
        self.u += (dt/2.0)*(self.rhs(self.u, self.t,*args)+self.rhs(u_1,self.t+dt, *args))
        #return self.u
    
    def get_Schemes(self) : 
        return tuple(self.__SchemeList.keys())

    def reset_t(self):
        self.t = 0.0
