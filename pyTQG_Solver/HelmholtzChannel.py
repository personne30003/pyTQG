"""
Solveur de Helmholtz pour grille périodique selon x et avec CLs de Dirichlet selon y (les CLs de Neumann/Robin ne sont pas implémentées)
Résout une équation de la forme (dx^2 + dy^2 - alpha^2)psi=RHS
Validé du premier coup !!

"""
import numpy as np
import scipy
import Fourier
import Chebyshev
import sys
import time

class HelmholtzChannel:
    def __init__(self, Nx, Ny, x_bounds = (-np.pi, np.pi), y_bounds = (-1.0, 1.0), alpha2 = 1.0, print_func = None,
                 BC_y =('Dirichlet', 'Dirichlet')):
        self.Nx = Nx
        self.Ny = Ny
        #Vérifications
        if len(x_bounds) !=2:
            raise ValueError("x_bounds must contain 2 values")
        if len(y_bounds) !=2:
            raise ValueError("y_bounds must contain 2 values")

        if x_bounds[0] >= x_bounds[1]:
            raise ValueError("x_bounds's values must be sorted in ascending order")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError("y_bounds's values must be sorted in ascending order")

        if len(BC_y) != 2:
            raise ValueError("BC_y must contain 2 values")
        possible_BC = ('Dirichlet', 'Neumann')
        if (BC_y[0] not in possible_BC) or (BC_y[1] not in possible_BC):
            raise ValueError(f"BC_y's values must be in {possible_BC}, here BC_y = {BC_y}")
        if (BC_y[0] != 'Dirichlet') or (BC_y[1] != 'Dirichlet'):
            raise NotImplementedError("Only Dirichlet BC are implemented ...")
        #Au cas où on devrait afficher sur la sortie standard et dans un fichier LOG
        self.disp_func = print
        if print_func is not None:
            self.disp_func = disp_func

        self.disp_func("Solveur choisi : HelmholtzChannel")

        self.kx = 2.0*np.pi*scipy.fft.fftfreq(self.Nx, d=(x_bounds[1] - x_bounds[0])/self.Nx)
        self.D2 = Chebyshev.Cheb_mat(self.Ny, a = y_bounds[0], b=y_bounds[-1], Dirichlet_BC = False, M=2)
        self.alpha2 = alpha2
        #Pré-construction et factorisation des matrices
        self.liste_piv = [0 for i in range(0, self.Nx)]
        self.liste_F_fac = [0 for i in range(0, self.Nx)]

        alpha_Id = self.alpha2*np.eye(self.Ny - 2)
        self.disp_func("Calcul des matrices de différenciation et préfactorisation LU")
        t0=time.time()
        
        for i in range(0, self.Nx):
            F_i = np.zeros( (self.Ny-2, self.Ny-2) )
            np.fill_diagonal(F_i, -(self.alpha2+self.kx[i]**2))
            F_i += self.D2[1:-1, 1:-1].copy()

            lu_i, piv_i = scipy.linalg.lu_factor(F_i)
            self.liste_piv[i] = piv_i
            self.liste_F_fac[i] = lu_i
        t1 = time.time()
        self.disp_func(f"Fait en {(t1-t0):.3f} s")
        #Affiche le volume de mémoire consommé (en MiB)
        size_liste_mat = lambda liste_mat : (sys.getsizeof(liste_mat)+sum(mat.nbytes for mat in liste_mat))/(1024**2)
        size_prefac = size_liste_mat(self.liste_piv)+size_liste_mat(self.liste_F_fac)
        self.disp_func(f"Mémoire occupée par ces matrices : {size_prefac:.2f} MiB")

    def Solve(self, rhs, CL_y_inf, CL_y_sup, real = True):

        #Vérifications usuelles
        
        if np.isscalar(rhs) : 
            RHS = rhs *np.ones((self.Nx, self.Ny))
        elif isinstance(rhs, np.ndarray): 
            if (rhs.shape[0] != self.Nx) or (rhs.shape[1] != self.Ny):
                raise ValueError(f"rhs must have shape Nx*Ny (here rhs.shape = {rhs.shape})")
            RHS = rhs.copy()
        else:
            raise ValueError(f"rhs must be scalar or np.ndarray of size Nx x Ny")

        if np.isscalar(CL_y_inf) : 
            BC_y_inf = np.ones(self.Nx) * CL_y_inf
        elif ( isinstance(CL_y_inf, np.ndarray) == True ) and ( CL_y_inf.shape[0] == self.Nx and len(CL_y_inf.shape) == 1 ): 
            BC_y_inf = CL_y_inf.copy()
        else : 
            raise ValueError(f"CL_y_inf must be scalar or 1D array of size Nx (CL_y_inf = {CL_y_inf}")
        
        if np.isscalar(CL_y_sup) : 
            BC_y_sup = np.ones(self.Nx) * CL_y_sup
        elif ( isinstance(CL_y_sup, np.ndarray) == True ) and ( CL_y_sup.shape[0] == self.Nx and len(CL_y_sup.shape) == 1 ): 
            BC_y_sup = CL_y_sup.copy()
        else : 
            raise ValueError(f"CL_y_sup must be scalar or 1D array of size Nx (CL_y_sup = {CL_y_sup}")

        psi_num = np.zeros((self.Nx, self.Ny))
        psi_num_hat = np.zeros((self.Nx, self.Ny), dtype = 'complex')

        rhs_hat = scipy.fft.fft(RHS, axis = 0)

        BC_y_inf_hat = scipy.fft.fft(BC_y_inf)
        BC_y_sup_hat = scipy.fft.fft(BC_y_sup)

        self.disp_func("Appel HelmholtzChannel")
        t0 = time.time()

        for i in range(0, self.Nx):
            rhs_hat_i = rhs_hat[i, 1:-1].copy()
            coeff_BC_sup = self.D2[1:-1, -0].copy()
            coeff_BC_inf = self.D2[1:-1, -1].copy()
            rhs_hat_BC = rhs_hat_i - coeff_BC_sup*BC_y_sup_hat[i] - coeff_BC_inf*BC_y_inf_hat[i]
            psi_num_hat[i, 1:-1] = scipy.linalg.lu_solve((self.liste_F_fac[i], self.liste_piv[i]), rhs_hat_BC)
            #on ajoute les BC
            psi_num_hat[i, 0] = BC_y_sup_hat[i]
            psi_num_hat[i, -1] = BC_y_inf_hat[i]

        psi_num = scipy.fft.ifft(psi_num_hat, axis = 0)
        self.disp_func(f"Fait en {time.time()-t0:.3f} s")
        
        if real:
            return psi_num.real
        return psi_num