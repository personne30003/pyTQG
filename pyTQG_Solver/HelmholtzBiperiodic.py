"""
Solveur de Helmholtz pour grille bipériodique (Fourier-Fourier). 
Résout une équation de la forme (dx^2 + dy^2 - alpha^2)psi=RHS

Les "**kwargs", ce sont des arguments nommés, pas utilisés ici.
Ils sont juste là pour etre compatibles avec les autres solveurs

Validé !
"""


import numpy as np
import scipy
import warnings


class HelmholtzBiperiodic:
    def __init__(self,
                 Nx,
                 Ny,
                 x_bounds = (-np.pi, np.pi),
                 y_bounds = (-np.pi, np.pi),
                 kx = None,
                 ky =None,
                 **kwargs):
        self.Nx = Nx
        self.Ny = Ny

        if len(x_bounds) !=2:
            raise ValueError("x_bounds must contain 2 values")
        if len(y_bounds) !=2:
            raise ValueError("y_bounds must contain 2 values")

        if x_bounds[0] >= x_bounds[1]:
            raise ValueError("x_bounds's values must be sorted in ascending order")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError("y_bounds's values must be sorted in ascending order")


        if kx is None:
            Kx = 2.0*np.pi*scipy.fft.fftfreq(self.Nx, d = ( (x_bounds[1] - x_bounds[0])/self.Nx ) )
            #print(f"kx = {Kx}")
        else:
            Kx = kx

        if ky is None:
            y = np.linspace(y_bounds[0], y_bounds[1], self.Ny, endpoint=False)
            Ky = 2.0*np.pi*scipy.fft.fftfreq(self.Ny, d = ( (y_bounds[1] - y_bounds[0])/self.Ny ) )
            #print(f"ky = {Ky}")
        else:
            Ky = ky

        print("Elliptic solver choosen : HelmholtzBiperiodic")
        self.__KX2, self.__KY2 = np.meshgrid(Kx**2, Ky**2, indexing = 'ij')
        self.__dic_alpha2 = {}


    def Solve(self, rhs, alpha2, real = True, **kwargs):
        #Verifications (suppose que RHS est de type np.ndarray)
        if alpha2 not in self.__dic_alpha2.keys() and (type(alpha2) == float):
            pass
        elif alpha2 in self.__dic_alpha2.keys():
            alpha2 = self.__dic_alpha2[alpha2]
        else:
            raise KeyError(f"alpha2 = {alpha2} not float or key of __dic_alpha2 (keys {self.__dic_alpha2.keys()}")

        if alpha2 == 0.0:
            alpha2 = 1.0e-16#Pour éviter division par zéro
            
        if np.isscalar(rhs) : 
            RHS = rhs *np.ones((self.Nx, self.Ny))
        elif isinstance(rhs, np.ndarray): 
            if (rhs.shape[0] != self.Nx) or (rhs.shape[1] != self.Ny):
                raise ValueError(f"rhs must have shape Nx*Ny (here rhs.shape = {rhs.shape})")
            RHS = rhs.copy()
        else:
            raise ValueError(f"rhs must be scalar or np.ndarray of size Nx x Ny")

        #Resolution ...
        RHS_hat = scipy.fft.fft2(RHS, axes = (0, 1))
        Psi_hat = -RHS_hat/( self.__KX2 + self.__KY2 + alpha2 )
        Psi = scipy.fft.ifft2(Psi_hat, axes = (0, 1))

        if real : 
            return Psi.real
        return Psi

    def __repr__(self):
        return r"Elliptic Solver : HelhmholtzBiperiodic\n"
    @property
    def alpha2(self):
        return self.__dic_alpha2

    @alpha2.setter
    def alpha2(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"value must be list or tuple, not {type(value)}")
        if len(value) != 2:
            raise ValueError(f"value must have lenght of 2, not {len(value)}")
        if type(value[0]) not in (int, str):
            warnings.warn(f"type {type(value[0])} not recommended for dictionnary key, please use int or str instead",
                          RuntimeWarning)
        self.__dic_alpha2[value[0]] = value[1]