"""
Solveur de Helmholtz pour grille bipériodique (Fourier-Fourier). 
Résout une équation de la forme (dx^2 + dy^2 - alpha^2)psi=RHS

Les "**kwargs", ce sont des arguments nommés, pas utilisés ici. Ils sont juste là pour etre compatibles avec les autres solveurs

Non validé pour le moment |=-(
"""


import numpy as np
import scipy



class HelmholtzBiperiodic:
    def __init__(self, Nx, Ny, x_bounds = (-np.pi, np.pi), y_bounds = (-np.pi, np.pi),kx = None, ky =None, alpha2 = 1.0, print_func = None, **kwargs):
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
        
        #print(f"x_sup =x[-1] = {x[-1]}, x_inf =x[0] = {x[0]}")
        #print(f"y_sup =y[-1] = {y[-1]}, y_inf =y[0] = {y[0]}")
        #print(f"kx = {kx}")
        #print(f"ky = {ky}")
        self.KX2, self.KY2 = np.meshgrid(Kx**2, Ky**2, indexing = 'ij')
        self.alpha2 = alpha2
        if self.alpha2 == 0.0:
            self.alpha2 = 1.0e-16#Pour éviter division par zéro

    def Solve(self, rhs, real = True, **kwargs):
        #Verifications (suppose que RHS est de type np.ndarray)
        if np.isscalar(rhs) : 
            RHS = rhs *np.ones((self.Nx, self.Ny))
        elif isinstance(rhs, np.ndarray): 
            if (rhs.shape[0] != self.Nx) or (rhs.shape[1] != self.Ny):
                raise ValueError(f"rhs must have shape Nx*Ny (here rhs.shape = {rhs.shape})")
            RHS = rhs.copy()
        else:
            raise ValueError(f"rhs must be scalar or np.ndarray of size Nx x Ny")

        RHS_hat = scipy.fft.fft2(RHS, axes = (0, 1))
        Psi_hat = -RHS_hat/( self.KX2 + self.KY2 + self.alpha2 )
        Psi = scipy.fft.ifft2(Psi_hat, axes = (0, 1))

        if real : 
            return Psi.real
        return Psi