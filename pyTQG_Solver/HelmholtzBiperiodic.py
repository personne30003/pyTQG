"""
Solveur de Helmholtz pour grille bipériodique (Fourier-Fourier). 
Résout une équation de la forme (dx^2 + dy^2 - alpha^2)psi=RHS

Les "**kwargs", ce sont des arguments nommés, pas utilisés ici. Ils sont juste là pour etre compatibles avec les autres solveurs
"""


import numpy as np
import scipy



class HelmholtzBiperiodic:
    def __init__(self, Nx, Ny, x_bounds = (-np.pi, np.pi), y_bounds = (-np.pi, np.pi), alpha2 = 1.0, print_func = None, **kwargs):
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

        kx = 2.0*np.pi*scipy.fft.fftfreq(self.Nx, d = (x_bounds[1] - x_bounds[0])/self.Nx)
        ky = 2.0*np.pi*scipy.fft.fftfreq(self.Ny, d = (y_bounds[1] - y_bounds[0])/self.Ny)

        self.KX2, self.KY2 = np.meshgrid(kx**2, ky**2, indexing = 'ij')
        self.alpha2 = alpha2

    def Solve(self, rhs, real = True, **kwargs):
        #Verifications (suppose que RHS est de type np.ndarray)
        RHS_ = None
        if np.isscalar(rhs):
            RHS_ = np.ones(rhs)
        if isinstance(rhs, np.ndarray) and (rhs.shape[0] == Nx or rhs.shape[1] == Ny) : 
            RHS_ = rhs.copy()
        else:
            raise ValueError(f"rhs must be scalar or np.ndarray of size Nx x Ny (rhs.shape = {rhs.shape})")

        RHS_hat = scipy.fft.fft2(RHS_)
        Psi_hat = -RHS_hat/( self.KX2 + self.KY2 + self.alpha2 )
        Psi = scipy.fft.ifft2(Psi_hat)

        if real : 
            return Psi.real
        return Psi